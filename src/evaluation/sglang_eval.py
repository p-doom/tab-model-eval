import asyncio
import json
import os
import re
import sys
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import httpx
import tyro
from openai import AsyncOpenAI, BadRequestError
from tqdm.asyncio import tqdm_asyncio


# ----------------------------
# Argument definitions
# ----------------------------


@dataclass
class Args:
    # Eval-related
    input_file: str = "data/eval/hello_world_insert_generations.jsonl"
    output_file: str = "data/eval/hello_world_insert_evaluations.jsonl"
    limit: int = -1
    prompt_file: str = "data/prompts/command_evaluation_prompt.txt"
    judge_name: str = "default"
    accept_threshold: float = 0.8

    # Server-related (sglang)
    model_path: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    server_host: str = "0.0.0.0"
    server_port: int = 30000
    context_length: int = 128000
    mem_fraction_static: float = 0.95

    # HTTP / client config
    concurrency: int = 64
    max_connections: int = 256
    keepalive: int = 60
    max_attempts: int = 6
    timeout: float = 30.0

    # Control whether to launch server from this script
    launch_server: bool = True
    # Extra args passed to `sglang.launch_server` if needed
    extra_server_args: Optional[List[str]] = None


# ----------------------------
# Dataset helpers
# ----------------------------


def load_dataset(filepath):
    """Yields each test case from the jsonl file."""
    with open(filepath, "r") as f:
        for line in f:
            yield json.loads(line)


# ----------------------------
# Eval logic
# ----------------------------


async def evaluate_generated_command(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    test_case: Dict[str, Any],
    args: Args,
) -> Dict[str, Any]:
    """
    Handles a single evaluation task with concurrency control and retries.
    """
    async with sem:
        delay = 0.25

        if test_case.get("error", None) is not None:
            print(
                f"Returning failure object for task {test_case['task_id']} due to error"
            )
            return {
                "task_id": test_case["task_id"],
                "error": test_case["error"],
                "is_correct": 0,
                "average_score": 0.0,
            }

        for attempt in range(args.max_attempts):
            try:
                with open(args.prompt_file, "r") as pf:
                    prompt_template = pf.read()

                prompt = prompt_template.format(
                    context=json.dumps(test_case["context"], indent=2),
                    expected=test_case["expected_command"],
                    generated=test_case["generated_command"],
                )

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert command evaluator specializing in "
                            "semantic equivalence and correctness assessment."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]

                resp = await client.chat.completions.create(
                    model=args.judge_name,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                result = json.loads(resp.choices[0].message.content)

                semantic_eq = result.get("semantic_equivalence", {})
                correctness = result.get("correctness", {})
                average_score = (
                    semantic_eq.get("score", 0.0) + correctness.get("score", 0.0)
                ) / 2.0
                is_correct = 1 if average_score >= args.accept_threshold else 0

                return {
                    "task_id": test_case["task_id"],
                    "generated_command": test_case["generated_command"],
                    "expected_command": test_case["expected_command"],
                    "average_score": average_score,
                    "evaluation_results": result,
                    "is_correct": is_correct,
                }

            except BadRequestError as e:
                print(
                    f"Returning failure object for task {test_case['task_id']} due to BadRequestError: {e}"
                )
                return {
                    "task_id": test_case["task_id"],
                    "error": str(e),
                    "is_correct": 0,
                    "average_score": 0.0,
                }

            except Exception as e:
                print(f"Error on task {test_case['task_id']}: {e}")
                if attempt == args.max_attempts - 1:
                    print(f"Returning failure object for task {test_case['task_id']}")
                    return {
                        "task_id": test_case["task_id"],
                        "error": str(e),
                        "is_correct": 0,
                        "average_score": 0.0,
                    }
                await asyncio.sleep(delay)
                delay *= 2


async def run_eval(args: Args, base_url: str):
    test_cases = list(load_dataset(args.input_file))
    if args.limit > 0:
        test_cases = test_cases[: args.limit]

    # Clean output
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    # Reuse a single HTTP/2 client with a large pool
    http = httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(args.timeout, connect=10.0, read=args.timeout),
        limits=httpx.Limits(
            max_connections=args.max_connections,
            max_keepalive_connections=args.max_connections,
            keepalive_expiry=args.keepalive,
        ),
        headers={"Connection": "keep-alive"},
    )
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="EMPTY",  # sglang OpenAI-compatible server usually ignores this
        http_client=http,
    )

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [evaluate_generated_command(client, sem, tc, args) for tc in test_cases]

    print(
        f"Running {len(test_cases)} test cases with concurrency={args.concurrency} ..."
    )
    results: List[Dict[str, Any]] = []

    # progress bar over async tasks
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        results.append(await coro)

    # sort the results by task_id
    results.sort(key=lambda x: x["task_id"])

    num_correct = sum(r.get("is_correct", 0) for r in results)
    total_score_sum = sum(r.get("average_score", 0.0) for r in results)

    # Write once
    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    pass_rate = 100.0 * num_correct / len(test_cases) if test_cases else 0.0
    avg_score = total_score_sum / len(test_cases) if test_cases else 0.0

    print("\n" + "=" * 50)
    print("--- Evaluation Complete ---")
    print("=" * 50)
    print(f"Total Test Cases: {len(test_cases)}")
    print(f"Correct (threshold >= {args.accept_threshold}): {num_correct}")
    print(f"Pass Rate: {pass_rate:.2f}%")
    print(f"Average Score: {avg_score:.2f}")

    await http.aclose()


# ----------------------------
# Server launch + waiting
# ----------------------------


async def wait_for_server(base_url: str, timeout: float = 120.0) -> None:
    """
    Poll the server's OpenAI-compatible endpoint until it responds or timeout.
    We’ll try a lightweight call to /models.
    """
    print(f"Waiting for server at {base_url} ...")
    deadline = asyncio.get_event_loop().time() + timeout

    async with httpx.AsyncClient() as client:
        while True:
            now = asyncio.get_event_loop().time()
            if now > deadline:
                raise RuntimeError(
                    f"Server at {base_url} did not become ready within {timeout} seconds."
                )
            try:
                resp = await client.get(f"{base_url}/v1/models", timeout=5.0)
                if resp.status_code == 200:
                    print("Server is up.")
                    return
                else:
                    print(
                        f"Server not ready yet (status {resp.status_code}); retrying..."
                    )
            except Exception as e:
                print(f"Server not ready yet ({e}); retrying...")
            await asyncio.sleep(10.0)


def launch_sglang_server(args: Args) -> subprocess.Popen:
    """
    Launch sglang server as a subprocess.
    You should have `module load CUDA/12.8` and `source .venv/bin/activate`
    done in your shell before running this script.
    """
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--host",
        args.server_host,
        "--port",
        str(args.server_port),
        "--context-length",
        str(args.context_length),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
    ]

    if args.extra_server_args:
        cmd.extend(args.extra_server_args)

    print("Launching sglang server:")
    print("  " + " ".join(cmd))

    env = os.environ.copy()
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return proc


# ----------------------------
# Main
# ----------------------------


async def amain(args: Args):
    base_url = f"http://{args.server_host}:{args.server_port}/v1"
    print(f"Using server at {base_url}")

    server_proc: Optional[subprocess.Popen] = None
    try:
        if args.launch_server:
            server_proc = launch_sglang_server(args)
            await wait_for_server(f"http://{args.server_host}:{args.server_port}")

        await run_eval(args, base_url=base_url)

    finally:
        if server_proc is not None:
            print("Shutting down sglang server ...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print("Server did not exit in time; killing.")
                server_proc.kill()


if __name__ == "__main__":
    args = tyro.cli(Args)
    asyncio.run(amain(args))
    print("Done")
