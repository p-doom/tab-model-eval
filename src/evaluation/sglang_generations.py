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
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


# ----------------------------
# Argument definitions
# ----------------------------
@dataclass
class Args:
    # Eval-related
    test_cases_file: str = (
        "data/eval/handcrafted_test_cases/handcrafted_test_cases.jsonl"
    )
    generations_file: str = (
        "data/eval/handcrafted_test_cases/handcrafted_generations.jsonl"
    )
    limit: int = -1
    system_prompt_file: str = "data/prompts/minimal_v1.md"
    model_name: str = "default"

    # Server-related (sglang)
    model_path: str = "Qwen/Qwen3-0.6B"
    server_host: str = "0.0.0.0"
    server_port: int = 30000
    context_length: int = 40960
    problem_length: int = 40960
    mem_fraction_static: float = 0.95

    # HTTP / client config
    concurrency: int = 64
    max_connections: int = 256
    keepalive: int = 60
    max_attempts: int = 6
    timeout: float = 300.0

    # Control whether to launch server from this script
    launch_server: bool = True
    # Extra args passed to `sglang.launch_server` if needed
    extra_server_args: Optional[List[str]] = None


# ----------------------------
# Dataset helpers
# ----------------------------
def load_dataset(filepath):
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
        return data


def extract_first_bash_block(text: str) -> str:
    m = re.search(r"(```bash\s+.*?```)", text, re.DOTALL)
    return m.group(1) if m else ""


def estimate_token_count(messages: List[Dict[str, str]]) -> int:
    """
    Rough estimate of token count for a list of messages.
    Assumes ~3 characters per token as a conservative estimate.
    """
    total_chars = sum(len(msg.get("content", "")) for msg in messages)
    return total_chars // 3


def filter_tasks_by_context_length(
    test_cases: List[Dict[str, Any]],
    system_prompt: str,
    max_context_length: int = 40960,
    problem_length: int = 40960,
    buffer_tokens: int = 512,  # Reserve space for response
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter out test cases whose context would exceed the model's context length.
    Returns (valid_cases, skipped_cases)
    """
    valid_cases = []
    skipped_cases = []

    for tc in test_cases:
        # Estimate tokens for system prompt + context
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(tc["context"])
        estimated_tokens = estimate_token_count(messages)

        length = estimated_tokens + buffer_tokens
        if length <= max_context_length and length <= problem_length:
            valid_cases.append(tc)
        else:
            print(
                f"Skipping {tc['task_id']}: estimated {estimated_tokens} tokens (limit: {max_context_length}, problem_length: {problem_length})"
            )
            skipped_cases.append(
                {
                    "task_id": tc["task_id"],
                    "estimated_tokens": estimated_tokens,
                    "reason": "context_too_long",
                }
            )

    return valid_cases, skipped_cases


# ----------------------------
# Eval logic
# ----------------------------
async def generate_next_command(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    system_prompt: str,
    test_case: Dict[str, Any],
    model: str,
    max_attempts: int,
) -> Dict[str, Any]:
    formatted_messages = [{"role": "system", "content": system_prompt}]
    formatted_messages.extend(test_case["context"])

    async with sem:
        delay = 0.25
        for attempt in range(max_attempts):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=formatted_messages,
                    temperature=0.0,
                    stop=["ASSISTANT:", "USER:"],
                )
                response_text = resp.choices[0].message.content or ""
                generated = extract_first_bash_block(response_text)
                expected = test_case.get("expected_final_response", "")
                exact_match = int(generated == expected)
                return {
                    "task_id": test_case["task_id"],
                    "response_text": response_text,
                    "context": test_case["context"],
                    "generated_command": generated,
                    "expected_command": expected,
                    "exact_match": exact_match,
                }
            except Exception as e:
                print(f"Error on task {test_case['task_id']}: {e}")
                if attempt == max_attempts - 1:
                    print(f"Returning failure object for task {test_case['task_id']}")
                    return {
                        "task_id": test_case["task_id"],
                        "response_text": "",
                        "context": test_case["context"],
                        "generated_command": "",
                        "expected_command": test_case.get(
                            "expected_final_response", ""
                        ),
                        "exact_match": 0,
                        "error": str(e),
                    }
                await asyncio.sleep(delay)
                delay *= 2

        return {
            "task_id": test_case["task_id"],
            "error": "Max attempts reached",
            "is_correct": 0,
            "average_score": 0.0,
        }


async def run_eval(args: Args, base_url: str):
    test_cases = list(load_dataset(args.test_cases_file))
    if args.limit > 0:
        test_cases = test_cases[: args.limit]

    with open(args.system_prompt_file, "r") as f:
        system_prompt = f.read()

    # Filter out tasks with context that's too long
    test_cases, skipped_cases = filter_tasks_by_context_length(
        test_cases,
        system_prompt,
        max_context_length=args.context_length,
        problem_length=args.problem_length,
        buffer_tokens=512,
    )

    print(f"\nFiltered dataset:")
    print(f"  Valid test cases: {len(test_cases)}")
    print(f"  Skipped (too long): {len(skipped_cases)}")
    print()

    # Clean output
    if os.path.exists(args.generations_file):
        os.remove(args.generations_file)

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
        api_key="EMPTY",  # sglang’s OpenAI-compatible server usually ignores this
        http_client=http,
    )

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        generate_next_command(
            client, sem, system_prompt, tc, args.model_name, args.max_attempts
        )
        for tc in test_cases
    ]

    print(
        f"Running {len(test_cases)} test cases with concurrency={args.concurrency} ..."
    )
    results: List[Dict[str, Any]] = []
    # progress bar over async tasks
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        results.append(await coro)

    # sort the results by task_id
    results.sort(key=lambda x: x["task_id"])

    # Write once
    os.makedirs(os.path.dirname(args.generations_file), exist_ok=True)
    correct = sum(r.get("exact_match", 0) for r in results)
    pass_rate = 100.0 * correct / len(test_cases) if test_cases else 0.0
    with open(args.generations_file, "w") as f:
        json.dump(
            {
                "config_generations": args.__dict__,
                "generation_scores": {
                    "total_test_cases": len(test_cases),
                    "num_exact_match": correct,
                    "pass_rate": pass_rate,
                },
                "generation_results": results,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 50)
    print("--- Evaluation Complete ---")
    print("=" * 50)
    print(f"Total Test Cases: {len(test_cases)}")
    print(f"Correct (exact match): {correct}")
    print(f"Pass Rate: {pass_rate:.2f}%")

    await http.aclose()


# ----------------------------
# Server launch + waiting
# ----------------------------
async def wait_for_server(base_url: str, timeout: float = 300.0) -> None:
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
                # OpenAI-compatible /models endpoint
                resp = await client.get(f"{base_url}/v1/models", timeout=5.0)
                if resp.status_code == 200:
                    print("Server is up.")
                    return
                else:
                    print(
                        f"Server not ready yet (status {resp.status_code}); retrying..."
                    )
            except Exception as e:
                # Connection error or similar
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

    # Inherit current environment (which should already have CUDA + venv)
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
