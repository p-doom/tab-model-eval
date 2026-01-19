"""
LLM-as-a-judge evaluation for code completion models.

This script evaluates generated commands using an LLM judge to determine
semantic equivalence between generated and expected commands.

This is designed to be run AFTER sglang_eval_metrics.py, which handles
fast deterministic metrics. The judge can optionally skip samples that
failed format validation to save compute.

Usage:
    # Run on raw generations (will do format check inline)
    python sglang_eval_judge.py \
        --generations-file data/eval/output/generations.json \
        --evaluations-file data/eval/output/evaluations.json \
        --eval-step 1000

    # Run on metrics output (skip format-invalid samples)
    python sglang_eval_judge.py \
        --metrics-file data/eval/output/metrics.json \
        --evaluations-file data/eval/output/evaluations.json \
        --eval-step 1000 \
        --skip-format-invalid
"""

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
import tyro
import wandb
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from sglang_eval_utils import (
    LocalLogger,
    check_command_format,
    filter_tasks_by_context_length,
    load_dataset,
    save_dataset,
)


# ----------------------------
# Argument definitions
# ----------------------------
@dataclass
class Args:
    # Wandb logging
    wandb_project: str = "llm-coding-agent"
    wandb_name: str = "validation_set_judge"
    wandb_eval_type: str = "next_action_validation_set"
    wandb_tags: list[str] = field(default_factory=lambda: ["val_mini", "judge_eval"])
    wandb_id: str | None = None
    wandb_group: str = "debug"

    # Input files - either generations_file OR metrics_file (from sglang_eval_metrics.py)
    generations_file: str = ""
    metrics_file: str = ""  # Output from sglang_eval_metrics.py
    evaluations_file: str = ""
    eval_step: int = 0

    # Batch mode
    generations_files: str = ""
    metrics_files: str = ""
    evaluations_files: str = ""
    eval_steps: str = ""

    # Skip samples that failed format validation (only works with metrics_file input)
    skip_format_invalid: bool = True

    limit: int = -1
    system_prompt_file: str = "data/prompts/judge_system_prompt_v3.md"
    judge_prompt_file: str = "data/prompts/judge_prompt_v3.md"
    judge_prompt_file_with_context: str = "data/prompts/judge_prompt_v3_with_context.md"
    include_context: bool = True

    # Local logging for offline mode
    use_local_logger: bool = False
    local_log_dir: str = "data/eval/local_logs"

    # Server-related (sglang)
    judge_model_path: str = "Qwen/Qwen3-32B"
    judge_name: str = "default"
    server_host: str = "0.0.0.0"
    server_port: int = 30000
    context_length: int = 40960
    problem_length: int = 40960
    api_key: str = "EMPTY"
    mem_fraction_static: float = 0.95
    tp_size: int = 1

    # We set presence_penalty to 0.0 to avoid the model from hallucinating variable names.
    presence_penalty: float = 0.0
    num_samples: int = 1
    enable_thinking: bool = True

    # HTTP / client config
    concurrency: int = 16
    max_connections: int = 256
    keepalive: int = 60
    max_attempts: int = 6
    timeout: float = 30.0

    # Server control
    launch_server: bool = True
    extra_server_args: Optional[List[str]] = None

    def get_eval_jobs(self) -> List[tuple[str, str, str, int]]:
        """
        Returns a list of (input_file, input_type, evaluations_file, eval_step) tuples.
        input_type is either 'generations' or 'metrics'
        """
        # Check batch mode
        if self.evaluations_files and self.eval_steps:
            eval_files = [f.strip() for f in self.evaluations_files.split(",") if f.strip()]
            steps = [int(s.strip()) for s in self.eval_steps.split(",") if s.strip()]

            # Prefer metrics_files if provided
            if self.metrics_files:
                input_files = [f.strip() for f in self.metrics_files.split(",") if f.strip()]
                input_type = "metrics"
            elif self.generations_files:
                input_files = [f.strip() for f in self.generations_files.split(",") if f.strip()]
                input_type = "generations"
            else:
                raise ValueError("Batch mode requires either metrics_files or generations_files")

            if not (len(input_files) == len(eval_files) == len(steps)):
                raise ValueError(
                    f"Batch mode requires equal-length lists: "
                    f"input_files ({len(input_files)}), "
                    f"evaluations_files ({len(eval_files)}), "
                    f"eval_steps ({len(steps)})"
                )

            return [(f, input_type, e, s) for f, e, s in zip(input_files, eval_files, steps)]

        # Single file mode
        if self.evaluations_file:
            if self.metrics_file:
                return [(self.metrics_file, "metrics", self.evaluations_file, self.eval_step)]
            elif self.generations_file:
                return [
                    (self.generations_file, "generations", self.evaluations_file, self.eval_step)
                ]

        raise ValueError("Provide either metrics_file or generations_file, plus evaluations_file")


# ----------------------------
# Data loading helpers
# ----------------------------
def load_test_cases_from_generations(filepath: str) -> tuple[List[Dict], Dict, Dict]:
    """Load test cases from a generations file."""
    data = load_dataset(filepath)
    test_cases = data["generation_results"]
    config = data.get("config_generations", {})
    scores = data.get("generation_scores", {})
    return test_cases, config, scores


def load_test_cases_from_metrics(filepath: str) -> tuple[List[Dict], Dict, Dict]:
    """Load test cases from a metrics file (output of sglang_eval_metrics.py)."""
    data = load_dataset(filepath)
    test_cases = data["metrics_results"]
    config = data.get("metadata", {}).get("config_generations", {})
    scores = data.get("metrics_scores", {})
    return test_cases, config, scores


# ----------------------------
# Judge evaluation logic
# ----------------------------
async def evaluate_single_sample_with_judge(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    test_case: Dict[str, Any],
    sample: Dict[str, Any],
    sample_idx: int,
    args: Args,
    system_prompt: str,
    prompt_template: str,
) -> List[Dict[str, Any]]:
    """
    Evaluate a single sample using the LLM judge.
    Returns a list of evaluation results (one per judge sample).
    """
    async with sem:
        delay = 0.25
        results = []
        expected_command = test_case.get("expected_command", "")
        generated_command = sample.get("generated_command", "")

        for attempt in range(args.max_attempts):
            try:
                format_dict = {
                    "expected": expected_command,
                    "generated": generated_command,
                }
                if args.include_context:
                    format_dict["context"] = json.dumps(test_case.get("context", []), indent=2)
                prompt = prompt_template.format(**format_dict)

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]

                resp = await client.chat.completions.create(
                    model=args.judge_name,
                    messages=messages,
                    presence_penalty=args.presence_penalty,
                    n=args.num_samples,
                    response_format={"type": "json_object"},
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
                    },
                )

                for choice_idx, choice in enumerate(resp.choices):
                    thinking_trace = getattr(choice.message, "reasoning_content", "")
                    result = json.loads(choice.message.content)
                    equivalent = result.get("equivalent", 0)

                    results.append(
                        {
                            "sample_idx": sample_idx,
                            "choice_idx": choice_idx,
                            "messages": messages,
                            "generated_command": generated_command,
                            "thinking_trace": thinking_trace,
                            "evaluation_results": result,
                            "equivalent": equivalent,
                            "exact_match": sample.get("exact_match", 0),
                        }
                    )
                return results

            except Exception as e:
                print(f"Error on task {test_case['task_id']}: {e}")
                if attempt == args.max_attempts - 1:
                    return [
                        {
                            "sample_idx": sample_idx,
                            "choice_idx": None,
                            "task_id": test_case["task_id"],
                            "error": str(e),
                            "equivalent": 0,
                        }
                    ]
                await asyncio.sleep(delay)
                delay *= 2

        return results


async def evaluate_task_with_judge(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    test_case: Dict[str, Any],
    args: Args,
    system_prompt: str,
    prompt_template: str,
    input_type: str,
) -> Dict[str, Any]:
    """
    Evaluate all samples for a test case using the LLM judge.
    """
    # Handle error cases
    if test_case.get("error") or test_case.get("had_error"):
        return {
            "task_id": test_case["task_id"],
            "error": test_case.get("error", "Unknown error"),
            "had_error": True,
            "judge_avg_at_n": 0.0,
            "judge_pass_at_n": 0,
            "sample_evaluations": [],
        }

    samples = test_case.get("samples", [])
    if not samples:
        return {
            "task_id": test_case["task_id"],
            "error": "No samples",
            "had_error": True,
            "judge_avg_at_n": 0.0,
            "judge_pass_at_n": 0,
            "sample_evaluations": [],
        }

    expected_command = test_case.get("expected_command", "")

    # Build list of samples to evaluate
    samples_to_eval = []
    skipped_samples = []

    for idx, sample in enumerate(samples):
        should_skip = False
        skip_reason = None

        generated_command = sample.get("generated_command", "")
        if generated_command == "":
            should_skip = True
            skip_reason = "empty_generated_command"
        elif sample.get("exact_match", 0) == 1:
            should_skip = True
        elif args.skip_format_invalid and input_type == "metrics":
            if not sample.get("format_valid", True):
                should_skip = True
                skip_reason = sample.get("format_reason", "format_invalid")
        elif args.skip_format_invalid and input_type == "generations":
            format_valid, format_reason = check_command_format(generated_command, expected_command)
            if not format_valid:
                should_skip = True
                skip_reason = format_reason

        if should_skip:
            exact_match = sample.get("exact_match", 0)
            skipped_samples.append(
                {
                    "sample_idx": idx,
                    "generated_command": generated_command,
                    "equivalent": exact_match,
                    "skipped": True,
                    "skip_reason": skip_reason,
                    "exact_match": exact_match,
                }
            )
        else:
            samples_to_eval.append((idx, sample))

    if samples_to_eval:
        eval_tasks = [
            evaluate_single_sample_with_judge(
                client,
                sem,
                test_case,
                sample,
                idx,
                args,
                system_prompt,
                prompt_template,
            )
            for idx, sample in samples_to_eval
        ]
        eval_results_nested = await asyncio.gather(*eval_tasks)
        eval_results = []
        for results in eval_results_nested:
            eval_results.extend(results)
    else:
        eval_results = []

    all_results = skipped_samples + eval_results

    num_judge_matches = sum(r.get("equivalent", 0) for r in all_results if not r.get("skipped"))
    num_evaluated = len([r for r in all_results if not r.get("skipped")])
    num_skipped = len(skipped_samples)

    judge_avg_at_n = num_judge_matches / num_evaluated
    judge_pass_at_n = int(num_judge_matches > 0)

    return {
        "task_id": test_case["task_id"],
        "context": test_case.get("context", []),
        "expected_command": expected_command,
        "had_error": any("error" in r for r in all_results),
        "num_samples": len(samples),
        "num_evaluated": num_evaluated,
        "num_skipped": num_skipped,
        "num_judge_matches": num_judge_matches,
        "judge_avg_at_n": judge_avg_at_n,
        "judge_pass_at_n": judge_pass_at_n,
        "sample_evaluations": all_results,
    }


async def run_single_judge_eval(
    args: Args,
    input_file: str,
    input_type: str,
    evaluations_file: str,
    eval_step: int,
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    system_prompt: str,
    prompt_template: str,
    logger=None,
) -> Dict[str, Any]:
    """
    Run judge evaluation on a single file.
    """
    print(f"\n{'='*60}")
    print(f"Judge evaluating ({input_type}): {input_file}")
    print(f"Output: {evaluations_file}")
    print(f"Step: {eval_step}")
    print(f"Skip format invalid: {args.skip_format_invalid}")
    print(f"{'='*60}")

    if input_type == "metrics":
        test_cases, config_generations, gen_scores = load_test_cases_from_metrics(input_file)
    else:
        test_cases, config_generations, gen_scores = load_test_cases_from_generations(input_file)

    if args.limit > 0:
        test_cases = test_cases[: args.limit]

    test_cases, skipped_cases = filter_tasks_by_context_length(
        test_cases,
        system_prompt=system_prompt,
        prompt_template=prompt_template,
        max_context_length=args.context_length,
        problem_length=args.problem_length,
        buffer_tokens=512,
        include_context=args.include_context,
    )

    print(f"\nFiltered dataset:")
    print(f"  Valid test cases: {len(test_cases)}")
    print(f"  Skipped (too long): {len(skipped_cases)}")
    print()

    tasks = [
        evaluate_task_with_judge(client, sem, tc, args, system_prompt, prompt_template, input_type)
        for tc in test_cases
    ]

    print(f"Running judge on {len(test_cases)} test cases with concurrency={args.concurrency} ...")
    results: List[Dict[str, Any]] = []
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        results.append(await coro)

    # Sort by task_id
    results.sort(key=lambda x: x["task_id"])

    # Aggregate scores
    num_tasks = len(results)
    total_judge_matches = sum(r["num_judge_matches"] for r in results)
    total_evaluated = sum(r["num_evaluated"] for r in results)
    total_skipped = sum(r["num_skipped"] for r in results)

    avg_judge_at_n = sum(r["judge_avg_at_n"] for r in results) / num_tasks if num_tasks else 0.0
    total_judge_pass_at_n = sum(r["judge_pass_at_n"] for r in results)
    num_errors = sum(1 for r in results if r.get("had_error", False))

    gen_exact_match_avg = gen_scores.get(
        "total_exact_match_avg_at_n", gen_scores.get("gen_exact_match_avg_at_n", 0)
    )
    gen_exact_match_pass = gen_scores.get(
        "total_exact_match_pass_at_n", gen_scores.get("gen_exact_match_pass_at_n", 0)
    )

    scores = {
        "total_test_cases": num_tasks,
        "total_evaluated": total_evaluated,
        "total_skipped": total_skipped,
        "total_judge_matches": total_judge_matches,
        "avg_judge_at_n": avg_judge_at_n,
        "total_judge_pass_at_n": total_judge_pass_at_n,
        "gen_exact_match_avg_at_n": gen_exact_match_avg,
        "gen_exact_match_pass_at_n": gen_exact_match_pass,
        "num_errors": num_errors,
    }

    # Log metrics
    metrics_to_log = {
        "eval_step": eval_step,
        f"{args.wandb_eval_type}/total_test_cases": num_tasks,
        f"{args.wandb_eval_type}/total_evaluated": total_evaluated,
        f"{args.wandb_eval_type}/total_skipped": total_skipped,
        f"{args.wandb_eval_type}/avg_judge_at_n": avg_judge_at_n,
        f"{args.wandb_eval_type}/total_judge_pass_at_n": total_judge_pass_at_n,
        f"{args.wandb_eval_type}/gen_exact_match_avg_at_n": gen_exact_match_avg,
        f"{args.wandb_eval_type}/gen_exact_match_pass_at_n": gen_exact_match_pass,
        f"{args.wandb_eval_type}/num_errors": num_errors,
    }

    if args.use_local_logger and logger is not None:
        logger.log(metrics_to_log)
    else:
        wandb.log(metrics_to_log)

    # Save output
    output_data = {
        "metadata": {
            "config_generations": config_generations,
            "config_judge": args.__dict__,
            "eval_step": eval_step,
            "input_file": input_file,
            "input_type": input_type,
        },
        "judge_scores": scores,
        "judge_results": results,
    }
    save_dataset(evaluations_file, output_data)

    # Print summary
    print("\n" + "=" * 50)
    print(f"--- Judge Evaluation Complete (step {eval_step}) ---")
    print("=" * 50)
    print(f"Total Test Cases: {num_tasks}")
    print(f"Total Evaluated: {total_evaluated}")
    print(f"Total Skipped: {total_skipped}")
    print(f"Judge Avg@N: {avg_judge_at_n * 100:.2f}%")
    print(f"Judge Pass@N: {total_judge_pass_at_n}")
    print(f"Exact Match Avg@N: {gen_exact_match_avg * 100:.2f}%")
    print(f"Exact Match Pass@N: {gen_exact_match_pass}")
    print(f"Errors: {num_errors}")
    print(f"Output file: {evaluations_file}")

    return {
        "eval_step": eval_step,
        "input_file": input_file,
        "evaluations_file": evaluations_file,
        **scores,
    }


async def run_batch_judge_eval(args: Args, base_url: str):
    """
    Run judge evaluation on multiple files with a single model load.
    """
    eval_jobs = args.get_eval_jobs()

    print(f"\n{'#'*60}")
    print(f"# BATCH JUDGE EVALUATION")
    print(f"# Processing {len(eval_jobs)} evaluation job(s)")
    print(f"{'#'*60}\n")

    for i, (input_file, input_type, eval_file, step) in enumerate(eval_jobs):
        print(f"  [{i+1}/{len(eval_jobs)}] Step {step} ({input_type}): {input_file}")
    print()

    # Load prompts
    with open(args.system_prompt_file, "r") as f:
        system_prompt = f.read()

    judge_prompt_file = (
        args.judge_prompt_file_with_context if args.include_context else args.judge_prompt_file
    )
    with open(judge_prompt_file, "r") as f:
        prompt_template = f.read()

    # Initialize logger
    logger = None
    if args.use_local_logger:
        run_id = args.wandb_id or args.wandb_name
        logger = LocalLogger(
            log_dir=args.local_log_dir,
            run_id=run_id,
            run_name=args.wandb_name,
            project=args.wandb_project,
            eval_type=args.wandb_eval_type,
            config={"batch_mode": True, "num_jobs": len(eval_jobs)},
            tags=args.wandb_tags,
        )
    else:
        wandb_init_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_name,
            "tags": args.wandb_tags,
            "group": args.wandb_group,
            "config": {"batch_mode": True, "num_jobs": len(eval_jobs)},
        }
        if args.wandb_id:
            wandb_dir = os.path.join(os.getcwd(), "eval_logs", args.wandb_id)
            os.makedirs(wandb_dir, exist_ok=True)
            wandb_init_kwargs.update({"id": args.wandb_id, "resume": "allow", "dir": wandb_dir})
        wandb.init(**wandb_init_kwargs)

    # Create HTTP client
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
    client = AsyncOpenAI(base_url=base_url, api_key=args.api_key, http_client=http)
    sem = asyncio.Semaphore(args.concurrency)

    # Process each job
    all_results = []
    for i, (input_file, input_type, eval_file, step) in enumerate(eval_jobs):
        print(f"\n[{i+1}/{len(eval_jobs)}] Processing step {step}...")
        result = await run_single_judge_eval(
            args=args,
            input_file=input_file,
            input_type=input_type,
            evaluations_file=eval_file,
            eval_step=step,
            client=client,
            sem=sem,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
            logger=logger,
        )
        all_results.append(result)

    await http.aclose()

    # Finish logging
    if args.use_local_logger and logger is not None:
        logger.finish()
    else:
        wandb.finish()

    # Print summary
    print("\n" + "#" * 60)
    print("# BATCH JUDGE EVALUATION SUMMARY")
    print("#" * 60)
    for r in all_results:
        print(
            f"  Step {r['eval_step']:>5}: "
            f"Judge Avg@N = {r['avg_judge_at_n']*100:5.2f}%, "
            f"Pass@N = {r['total_judge_pass_at_n']}, "
            f"Evaluated = {r['total_evaluated']}, "
            f"Skipped = {r['total_skipped']}"
        )
    print("#" * 60)


# ----------------------------
# Server management
# ----------------------------
async def wait_for_server(base_url: str, timeout: float = 600.0) -> None:
    """Poll the server until it responds or timeout."""
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
    """Launch sglang server as a subprocess."""
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
    proc = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
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

        await run_batch_judge_eval(args, base_url=base_url)

    finally:
        if server_proc is not None:
            print("Shutting down sglang server ...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print("Server did not exit in time; killing.")
                server_proc.kill()


def main():
    args = tyro.cli(Args)
    asyncio.run(amain(args))
    print("Done")


if __name__ == "__main__":
    main()
