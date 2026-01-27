import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
import tyro
import yaml
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from .yaml_output import load_generation_yaml


JUDGE_SYSTEM_PROMPT = """You are an expert code reviewer evaluating tab completion model predictions.

Your task is to determine if the predicted code changes are semantically equivalent to the expected changes, based on the provided assertions.

Evaluate the prediction objectively:
1. Check if the predicted changes achieve the same functional outcome as expected
2. Consider the assertions provided - these describe what the change should accomplish
3. Minor formatting differences (whitespace, trailing newlines) should not affect the judgment
4. Variable naming differences are acceptable if semantically equivalent

Respond with a JSON object:
{
  "equivalent": 1 or 0,
  "reasoning": "Brief explanation of your judgment"
}
"""

JUDGE_PROMPT_TEMPLATE = """## Task
Evaluate if the predicted file changes are semantically equivalent to the expected changes.

## Assertions
{assertions}

## Expected Files
{expected_files}

## Predicted Files
{predicted_files}

## Instructions
Based on the assertions, determine if the predicted changes accomplish the same goal as the expected changes.
Respond with JSON: {{"equivalent": 1 or 0, "reasoning": "..."}}
"""


@dataclass
class Args:
    generations_file: str = "data/eval/generations/generations.yaml"
    evaluations_file: str = "data/eval/evaluations/evaluations.yaml"
    eval_step: int = 0
    limit: int = -1

    judge_model_path: str = "Qwen/Qwen3-32B"
    judge_name: str = "default"

    server_host: str = "0.0.0.0"
    server_port: int = 30000
    context_length: int = 40960
    mem_fraction_static: float = 0.95
    tp_size: int = 1
    api_key: str = "EMPTY"
    launch_server: bool = True
    extra_server_args: Optional[List[str]] = None

    presence_penalty: float = 0.0
    num_judge_samples: int = 1
    enable_thinking: bool = True

    concurrency: int = 16
    max_connections: int = 256
    max_attempts: int = 6
    timeout: float = 60.0

    skip_exact_matches: bool = True
    skip_empty_predictions: bool = True


def format_files_for_prompt(files: Dict[str, str]) -> str:
    if not files:
        return "(no files)"
    parts = []
    for path, content in files.items():
        parts.append(f"### {path}\n```\n{content}\n```")
    return "\n\n".join(parts)


async def judge_single_sample(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    task_id: str,
    sample_idx: int,
    expected_files: Dict[str, str],
    predicted_files: Optional[Dict[str, str]],
    assertions: Optional[str],
    args: Args,
) -> Dict[str, Any]:
    if predicted_files is None:
        return {
            "task_id": task_id,
            "sample_idx": sample_idx,
            "equivalent": 0,
            "skipped": True,
            "skip_reason": "no_prediction",
        }

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        assertions=assertions
        or "No specific assertions provided. Judge based on code correctness.",
        expected_files=format_files_for_prompt(expected_files),
        predicted_files=format_files_for_prompt(predicted_files),
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    async with sem:
        delay = 0.25
        for attempt in range(args.max_attempts):
            try:
                resp = await client.chat.completions.create(
                    model=args.judge_name,
                    messages=messages,
                    presence_penalty=args.presence_penalty,
                    n=args.num_judge_samples,
                    response_format={"type": "json_object"},
                    extra_body={"chat_template_kwargs": {"enable_thinking": args.enable_thinking}},
                )

                choice = resp.choices[0]
                thinking = getattr(choice.message, "reasoning_content", "")
                result = json.loads(choice.message.content or "{}")

                return {
                    "task_id": task_id,
                    "sample_idx": sample_idx,
                    "equivalent": result.get("equivalent", 0),
                    "reasoning": result.get("reasoning", ""),
                    "thinking": thinking,
                    "skipped": False,
                }

            except Exception as e:
                if attempt == args.max_attempts - 1:
                    return {
                        "task_id": task_id,
                        "sample_idx": sample_idx,
                        "equivalent": 0,
                        "error": str(e),
                        "skipped": False,
                    }
                await asyncio.sleep(delay)
                delay *= 2

    return {"task_id": task_id, "sample_idx": sample_idx, "equivalent": 0, "error": "max_attempts"}


async def evaluate_task(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    task_result: Dict[str, Any],
    args: Args,
) -> Dict[str, Any]:
    task_id = task_result.get("task_id", "unknown")
    states = task_result.get("states", [])
    samples = task_result.get("samples", [])

    expected_files = {}
    assertions = None
    for state in states:
        if state.get("eval") == "EVAL":
            expected_files = state.get("files", {})
            assertions = state.get("judge_assertions")
            break

    sample_evals = []
    tasks = []

    for sample in samples:
        sample_idx = sample.get("sample_idx", 0)
        predicted_files = sample.get("predicted_files")
        exact_match = sample.get("exact_match", 0)
        predicted_raw = sample.get("predicted_raw", "")

        if args.skip_exact_matches and exact_match == 1:
            sample_evals.append(
                {
                    "task_id": task_id,
                    "sample_idx": sample_idx,
                    "equivalent": 1,
                    "skipped": True,
                    "skip_reason": "exact_match",
                }
            )
            continue

        if args.skip_empty_predictions and not predicted_raw.strip():
            sample_evals.append(
                {
                    "task_id": task_id,
                    "sample_idx": sample_idx,
                    "equivalent": 0,
                    "skipped": True,
                    "skip_reason": "empty_prediction",
                }
            )
            continue

        tasks.append(
            judge_single_sample(
                client, sem, task_id, sample_idx, expected_files, predicted_files, assertions, args
            )
        )

    if tasks:
        judged = await asyncio.gather(*tasks)
        sample_evals.extend(judged)

    sample_evals.sort(key=lambda x: x.get("sample_idx", 0))

    num_samples = len(sample_evals)
    num_equivalent = sum(1 for s in sample_evals if s.get("equivalent", 0) == 1)
    num_skipped = sum(1 for s in sample_evals if s.get("skipped", False))

    return {
        "task_id": task_id,
        "assertions": assertions,
        "num_samples": num_samples,
        "num_equivalent": num_equivalent,
        "num_skipped": num_skipped,
        "num_judged": num_samples - num_skipped,
        "equivalent_rate": num_equivalent / num_samples if num_samples > 0 else 0.0,
        "pass_at_1": int(num_equivalent > 0),
        "sample_evaluations": sample_evals,
    }


async def run_judge_eval(args: Args, base_url: str):
    print(f"\n{'='*60}")
    print(f"Judge Evaluation: {args.generations_file}")
    print(f"Output: {args.evaluations_file}")
    print(f"{'='*60}")

    data = load_generation_yaml(args.generations_file)
    results = data.get("results", [])
    config = data.get("config", {})

    if args.limit > 0:
        results = results[: args.limit]

    print(f"Evaluating {len(results)} tasks with judge...")

    http = httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(args.timeout, connect=10.0, read=args.timeout),
        limits=httpx.Limits(
            max_connections=args.max_connections, max_keepalive_connections=args.max_connections
        ),
    )
    client = AsyncOpenAI(base_url=base_url, api_key=args.api_key, http_client=http)
    sem = asyncio.Semaphore(args.concurrency)

    tasks = [evaluate_task(client, sem, r, args) for r in results]

    task_evals: List[Dict[str, Any]] = []
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        task_evals.append(await coro)

    await http.aclose()

    task_evals.sort(key=lambda x: x["task_id"])

    num_tasks = len(task_evals)
    total_samples = sum(t["num_samples"] for t in task_evals)
    total_equivalent = sum(t["num_equivalent"] for t in task_evals)
    total_pass_at_1 = sum(t["pass_at_1"] for t in task_evals)
    total_skipped = sum(t["num_skipped"] for t in task_evals)
    total_judged = sum(t["num_judged"] for t in task_evals)

    avg_equivalent_rate = (
        sum(t["equivalent_rate"] for t in task_evals) / num_tasks if num_tasks > 0 else 0.0
    )

    scores = {
        "total_tasks": num_tasks,
        "total_samples": total_samples,
        "total_equivalent": total_equivalent,
        "total_pass_at_1": total_pass_at_1,
        "total_skipped": total_skipped,
        "total_judged": total_judged,
        "avg_equivalent_rate": avg_equivalent_rate,
        "pass_at_1_rate": total_pass_at_1 / num_tasks if num_tasks > 0 else 0.0,
    }

    output = {
        "metadata": {
            "generations_file": args.generations_file,
            "eval_step": args.eval_step,
            "judge_model": args.judge_model_path,
            "config": config,
        },
        "judge_scores": scores,
        "task_evaluations": task_evals,
    }

    os.makedirs(os.path.dirname(args.evaluations_file), exist_ok=True)
    with open(args.evaluations_file, "w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print("\n" + "=" * 50)
    print(f"Judge Evaluation Complete (step {args.eval_step})")
    print("=" * 50)
    print(f"Tasks: {num_tasks}")
    print(f"Samples: {total_samples} (judged: {total_judged}, skipped: {total_skipped})")
    print(f"Equivalent: {total_equivalent}/{total_samples} ({avg_equivalent_rate*100:.1f}%)")
    print(f"Pass@1: {total_pass_at_1}/{num_tasks} ({scores['pass_at_1_rate']*100:.1f}%)")
    print(f"Output: {args.evaluations_file}")

    return scores


async def wait_for_server(base_url: str, timeout: float = 300.0) -> None:
    print(f"Waiting for server at {base_url}...")
    deadline = asyncio.get_event_loop().time() + timeout

    async with httpx.AsyncClient() as client:
        while True:
            if asyncio.get_event_loop().time() > deadline:
                raise RuntimeError(f"Server not ready within {timeout}s")
            try:
                resp = await client.get(f"{base_url}/v1/models", timeout=5.0)
                if resp.status_code == 200:
                    print("Server is up.")
                    return
            except Exception as e:
                print(f"Waiting... ({e})")
            await asyncio.sleep(10.0)


def launch_sglang_server(args: Args) -> subprocess.Popen:
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

    print("Launching judge server: " + " ".join(cmd))
    return subprocess.Popen(cmd, env=os.environ.copy(), stdout=sys.stdout, stderr=sys.stderr)


async def amain(args: Args):
    base_url = f"http://{args.server_host}:{args.server_port}/v1"

    server_proc: Optional[subprocess.Popen] = None
    try:
        if args.launch_server:
            server_proc = launch_sglang_server(args)
            await wait_for_server(f"http://{args.server_host}:{args.server_port}")
        await run_judge_eval(args, base_url)
    finally:
        if server_proc:
            print("Shutting down server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_proc.kill()


def main():
    args = tyro.cli(Args)
    asyncio.run(amain(args))
    print("Done")


if __name__ == "__main__":
    main()
