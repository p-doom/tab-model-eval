import asyncio
import json
import os
import sys
import subprocess
import wandb
from dataclasses import dataclass, field
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
    wandb_project: str = "llm-coding-agent"
    wandb_name: str = "validation_set_eval"
    wandb_eval_type: str = "next_action_validation_set"
    wandb_tags: list[str] = field(default_factory=lambda: ["val_mini", "judge_eval"])
    generations_file: str = "data/eval/handcrafted_test_cases/handcrafted_generations.jsonl"
    evaluations_file: str = "data/eval/handcrafted_test_cases/handcrafted_evaluations.jsonl"
    limit: int = -1
    system_prompt_file: str = "data/prompts/judge_system_prompt_v2.md"
    judge_name: str = "default"
    judge_prompt_file: str = "data/prompts/judge_prompt_v2.md"
    judge_prompt_file_with_context: str = "data/prompts/judge_prompt_v2_with_context.md"

    # Server-related (sglang)
    judge_model_path: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    server_host: str = "0.0.0.0"
    server_port: int = 30000
    context_length: int = 40960
    problem_length: int = 40960
    api_key: str = "EMPTY"  # sglang’s OpenAI-compatible server ignores this value
    temperature: float = 0.7
    mem_fraction_static: float = 0.95
    tp_size: int = 1

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
    with open(filepath, "r") as f:
        return json.loads(f.read())


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
    prompt_template: str,
    max_context_length: int = 40960,
    problem_length: int = 40960,
    buffer_tokens: int = 512,  # Reserve space for response
    include_context: bool = False,
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
        if include_context:
            messages.extend(tc["context"])
        messages.append({"role": "user", "content": prompt_template})
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
async def evaluate_generated_command(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    test_case: Dict[str, Any],
    args: Args,
    system_prompt: str,
    prompt_template: str,
    include_context: bool,
) -> Dict[str, Any]:
    """
    Handles a single evaluation task with concurrency control and retries.
    """
    async with sem:
        delay = 0.25

        if test_case.get("error", None) is not None:
            print(f"Returning failure object for task {test_case['task_id']} due to error")
            return {
                "task_id": test_case["task_id"],
                "error": test_case["error"],
                "is_correct": 0,
                "average_score": 0.0,
            }

        samples = test_case.get("samples", [])
        if not samples:
            print(f"Returning failure object for task {test_case['task_id']} due to no samples")
            return {
                "task_id": test_case["task_id"],
                "error": "No samples",
                "is_correct": 0,
                "average_score": 0.0,
            }

        sample_results = []
        for sample in samples: 
            for attempt in range(args.max_attempts):
                try:
                    format_dict = {
                        "expected": test_case["expected_command"],
                        "generated": sample["generated_command"],
                    }
                    if include_context:
                        format_dict["context"] = json.dumps(test_case["context"], indent=2)
                    prompt = prompt_template.format(**format_dict)
    
                    messages = [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {"role": "user", "content": prompt},
                    ]
    
                    resp = await client.chat.completions.create(
                        model=args.judge_name,
                        messages=messages,
                        temperature=args.temperature,
                        response_format={"type": "json_object"},
                    )
                    result = json.loads(resp.choices[0].message.content)
    
                    equivalent = result.get("equivalent", 0)
    
                    sample_results.append({
                        "generated_command": sample["generated_command"],
                        "evaluation_results": result,
                        "equivalent": equivalent,
                    })
                    break
    
                except BadRequestError as e:
                    print(
                        f"Returning failure object for task {test_case['task_id']} due to BadRequestError: {e}"
                    )
                    sample_results.append({
                        "task_id": test_case["task_id"],
                        "error": str(e),
                        "equivalent": 0,
                    })
                    break
    
                except Exception as e:
                    print(f"Error on task {test_case['task_id']}: {e}")
                    if attempt == args.max_attempts - 1:
                        print(f"Returning failure object for task {test_case['task_id']}")
                        sample_results.append({
                            "task_id": test_case["task_id"],
                            "error": str(e),
                            "equivalent": 0,
                        })
                    await asyncio.sleep(delay)
                    delay *= 2


        # Compute avg@n and pass@n
        num_judge_matches = sum(s.get("equivalent", 0) for s in sample_results)
        judge_avg_at_n = num_judge_matches / len(sample_results)
        judge_pass_at_n = num_judge_matches > 0

        return {
            "task_id": test_case["task_id"],
            "context": test_case["context"],
            "expected_command": test_case["expected_command"],
            "sample_evaluations": sample_results,
            "num_samples": len(sample_results),
            "num_judge_matches": num_judge_matches,
            "judge_avg_at_n": judge_avg_at_n,
            "judge_pass_at_n": judge_pass_at_n,
        }

async def run_eval(args: Args, base_url: str):
    loaded_data = load_dataset(args.generations_file)
    test_cases = loaded_data["generation_results"]

    config_generations = loaded_data["config_generations"]
    config_evaluations = args.__dict__
    metadata = {
        "config_generations": config_generations,
        "config_evaluations": config_evaluations,
    }

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        tags=args.wandb_tags,
        config=metadata,
    )

    if args.limit > 0:
        test_cases = test_cases[: args.limit]

    with open(args.system_prompt_file, "r") as f:
        system_prompt = f.read()

    include_context = bool(args.judge_prompt_file_with_context)
    judge_prompt_file = args.judge_prompt_file_with_context or args.judge_prompt_file

    with open(judge_prompt_file, "r") as f:
        prompt_template = f.read()

    # Filter out tasks with context that's too long
    test_cases, skipped_cases = filter_tasks_by_context_length(
        test_cases,
        system_prompt=system_prompt,
        prompt_template=prompt_template,
        max_context_length=args.context_length,
        problem_length=args.problem_length,
        buffer_tokens=512,
        include_context=include_context,
    )

    print(f"\nFiltered dataset:")
    print(f"  Valid test cases: {len(test_cases)}")
    print(f"  Skipped (too long): {len(skipped_cases)}")
    print()

    # Clean output
    if os.path.exists(args.evaluations_file):
        os.remove(args.evaluations_file)

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
        api_key=args.api_key,
        http_client=http,
    )

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        evaluate_generated_command(
            client, sem, tc, args, system_prompt, prompt_template, include_context
        )
        for tc in test_cases
    ]

    print(f"Running {len(test_cases)} test cases with concurrency={args.concurrency} ...")
    results: List[Dict[str, Any]] = []

    # progress bar over async tasks
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        results.append(await coro)

    # sort the results by task_id
    results.sort(key=lambda x: x["task_id"])

    os.makedirs(os.path.dirname(args.evaluations_file), exist_ok=True)
    total_judge_avg_at_n = sum(r.get("judge_avg_at_n", 0) for r in results) / len(results)
    total_judge_pass_at_n = sum(r.get("judge_pass_at_n", 0) for r in results)

    total_exact_match_avg_at_n = loaded_data["generation_scores"]["total_exact_match_avg_at_n"]
    total_exact_match_pass_at_n = loaded_data["generation_scores"]["total_exact_match_pass_at_n"]
    wandb.log(
        {
            f"{args.wandb_eval_type}/total_test_cases": len(test_cases),
            f"{args.wandb_eval_type}/num_samples_per_task": loaded_data["config_generations"]["num_samples"],
            f"{args.wandb_eval_type}/total_judge_avg_at_n": total_judge_avg_at_n,
            f"{args.wandb_eval_type}/total_judge_pass_at_n": total_judge_pass_at_n,
            f"{args.wandb_eval_type}/total_exact_match_avg_at_n": total_exact_match_avg_at_n,
            f"{args.wandb_eval_type}/total_exact_match_pass_at_n": total_exact_match_pass_at_n,
        }
    )

    with open(args.evaluations_file, "w") as f:
        json.dump(
            {
                "metadata": metadata,
                "evaluation_scores": {
                    "total_test_cases": len(test_cases),
                    "num_samples_per_task": loaded_data["config_generations"]["num_samples"],
                    "total_judge_avg_at_n": total_judge_avg_at_n,
                    "total_judge_pass_at_n": total_judge_pass_at_n,
                    "total_exact_match_avg_at_n": total_exact_match_avg_at_n,
                    "total_exact_match_pass_at_n": total_exact_match_pass_at_n,
                    "max_attempts": args.max_attempts,
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
    print(f"Total Judge Pass At N: {total_judge_pass_at_n}")
    print(f"Total Judge Avg At N: {total_judge_avg_at_n * 100:.2f}%")
    print(f"Total Exact Match Pass At N: {total_exact_match_pass_at_n}")
    print(f"Total Exact Match Avg At N: {total_exact_match_avg_at_n * 100:.2f}%")
    print(f"Evaluations output file: {args.evaluations_file}")

    await http.aclose()
    wandb.finish()


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
                    print(f"Server not ready yet (status {resp.status_code}); retrying...")
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
        args.judge_model_path,
        "--host",
        args.server_host,
        "--port",
        str(args.server_port),
        "--context-length",
        str(args.context_length),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--tp-size",
        str(args.tp_size),
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
