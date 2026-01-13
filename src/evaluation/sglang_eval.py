import asyncio
import json
import os
import re
import sys
import subprocess
import time
import wandb
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

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
    wandb_id: str | None = None
    wandb_group: str = "debug"

    # Single-file mode (backward compatible)
    generations_file: str = ""
    evaluations_file: str = ""
    eval_step: int = 0

    # Batch mode: comma-separated lists of files and steps
    # When these are provided, they take precedence over single-file args
    generations_files: str = ""  # Comma-separated list of generation files
    evaluations_files: str = ""  # Comma-separated list of evaluation output files
    eval_steps: str = ""  # Comma-separated list of eval steps (integers)

    limit: int = -1
    system_prompt_file: str = "data/prompts/judge_system_prompt_v3.md"
    judge_name: str = "default"
    judge_prompt_file: str = "data/prompts/judge_prompt_v3.md"
    judge_prompt_file_with_context: str = "data/prompts/judge_prompt_v3_with_context.md"
    include_context: bool = True

    # Local logging for offline mode
    use_local_logger: bool = False
    local_log_dir: str = "data/eval/local_logs"

    # Server-related (sglang)
    judge_model_path: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    server_host: str = "0.0.0.0"
    server_port: int = 30000
    context_length: int = 40960
    problem_length: int = 40960
    api_key: str = "EMPTY"  # sglang's OpenAI-compatible server ignores this value
    mem_fraction_static: float = 0.95
    tp_size: int = 1

    # Client-related
    presence_penalty: float = 1.5
    num_samples: int = 1
    enable_thinking: bool = True

    # HTTP / client config
    concurrency: int = 16
    max_connections: int = 256
    keepalive: int = 60
    max_attempts: int = 6
    timeout: float = 30.0

    # Control whether to launch server from this script
    launch_server: bool = True
    # Extra args passed to `sglang.launch_server` if needed
    extra_server_args: Optional[List[str]] = None

    def get_eval_jobs(self) -> List[tuple[str, str, int]]:
        """
        Returns a list of (generations_file, evaluations_file, eval_step) tuples.
        If batch mode args are provided, uses those. Otherwise falls back to single-file mode.
        """
        if self.generations_files and self.evaluations_files and self.eval_steps:
            # Batch mode
            gen_files = [f.strip() for f in self.generations_files.split(",") if f.strip()]
            eval_files = [f.strip() for f in self.evaluations_files.split(",") if f.strip()]
            steps = [int(s.strip()) for s in self.eval_steps.split(",") if s.strip()]

            if not (len(gen_files) == len(eval_files) == len(steps)):
                raise ValueError(
                    f"Batch mode requires equal-length lists for generations_files ({len(gen_files)}), "
                    f"evaluations_files ({len(eval_files)}), and eval_steps ({len(steps)})"
                )

            return list(zip(gen_files, eval_files, steps))
        elif self.generations_file and self.evaluations_file:
            # Single-file mode (backward compatible)
            return [(self.generations_file, self.evaluations_file, self.eval_step)]
        else:
            raise ValueError(
                "Either provide single-file args (generations_file, evaluations_file) "
                "or batch args (generations_files, evaluations_files, eval_steps)"
            )


# ----------------------------
# Local logger, since wandb offline can't resume runs
# ----------------------------
class LocalLogger:
    """A simple local logger that saves metrics to JSON files for later sync to wandb."""

    def __init__(
        self,
        log_dir: str,
        run_id: str,
        run_name: str,
        project: str,
        config: dict = None,
        tags: list = None,
    ):
        self.log_dir = os.path.join(log_dir, run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        self.run_id = run_id
        self.run_name = run_name
        self.project = project
        self.config = config or {}
        self.tags = tags or []
        self.metrics_file = os.path.join(self.log_dir, "metrics.jsonl")

        # Save run metadata
        metadata_file = os.path.join(self.log_dir, "metadata.json")
        if os.path.exists(metadata_file):
            # Avoid overwriting existing metadata for the same run_id
            print(
                f"Metadata file already exists for run_id={run_id} at {metadata_file}. "
                f"Existing metadata will be reused."
            )
        else:
            with open(metadata_file, "w") as f:
                json.dump(
                    {
                        "run_id": run_id,
                        "run_name": run_name,
                        "project": project,
                        "config": config,
                        "tags": tags,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                    f,
                    indent=2,
                )
        print(f"LocalLogger initialized. Logs will be saved to: {self.log_dir}")

    def log(self, metrics: dict):
        """Append metrics to the JSONL file."""
        metrics_with_timestamp = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **metrics}
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics_with_timestamp) + "\n")
        print(f"Logged metrics to {self.metrics_file}: eval_step={metrics.get('eval_step', 'N/A')}")

    def finish(self):
        """Called when logging is complete."""
        print(f"LocalLogger finished. All logs saved to: {self.log_dir}")


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
# Command format validation
# ----------------------------
def check_command_format(command: str, expected_command: str) -> Tuple[bool, str]:
    r"""
    Validates that a command follows the required edit format.

    Only applies edit-specific validation when expected_command is a sed edit command.

    Returns (is_valid, reason):
    - If expected is not a sed edit: always valid, returns (True, "non_edit_task")
    - If expected is a sed edit, the generated command must:
      - Use one of the 4 allowed sed patterns (not s/old/new/ substitution)
      - Not use file redirection (>, >>, tee) to modify files

    Valid sed patterns:
      1. Replace block: sed -i 'START,ENDc\...'
      2. Delete block: sed -i 'START,ENDd'
      3. Insert before: sed -i 'STARTi\...'
      4. Append to end: sed -i '$a\...'
    """
    if "sed -i" not in expected_command:
        return (True, "non_edit_task")

    # Check for file redirection patterns (>, >>, tee) - these are not allowed for edits
    # Match patterns like: echo "x" > file.py, cat content >> file.txt, tee file.py
    file_redirection_pattern = r"(^|[;&|])\s*(echo|cat|printf)\s+.*\s*>{1,2}\s*\S+"
    if re.search(file_redirection_pattern, command):
        return (False, "invalid_file_redirection")

    # Check for tee command writing to files
    tee_pattern = r"(^|[;&|])\s*tee\s+"
    if re.search(tee_pattern, command):
        return (False, "invalid_tee_redirection")

    # Check for heredoc patterns: cat << EOF > file
    heredoc_pattern = r"<<\s*\w+.*>"
    if re.search(heredoc_pattern, command):
        return (False, "invalid_heredoc_redirection")

    # If generated command doesn't use sed -i at all, it's invalid for an edit task
    if "sed -i" not in command:
        return (False, "missing_sed_edit")

    # Check for forbidden s/old/new/ substitution pattern
    # This pattern matches things like: sed -i '3s/old/new/g' or sed -i "s/foo/bar/"
    substitution_pattern = r"sed\s+-i\s+['\"][^'\"]*\d*s/"
    if re.search(substitution_pattern, command):
        return (False, "invalid_sed_substitution")

    # Allowed edit patterns:
    # 1. Replace block: sed -i 'START,ENDc\...' (e.g., sed -i '5,10c\new content')
    # 2. Delete block: sed -i 'START,ENDd' (e.g., sed -i '5,10d')
    # 3. Insert before: sed -i 'STARTi\...' (e.g., sed -i '5i\new line')
    # 4. Append to end: sed -i '$a\...' (e.g., sed -i '$a\new line')

    # Pattern for replace block: 'NUMBER,NUMBERc\'
    replace_pattern = r"sed\s+-i\s+['\"](\d+),(\d+)c\\"

    # Pattern for delete block: 'NUMBER,NUMBERd'
    delete_pattern = r"sed\s+-i\s+['\"](\d+),(\d+)d['\"]"

    # Pattern for insert before: 'NUMBERi\'
    insert_pattern = r"sed\s+-i\s+['\"](\d+)i\\"

    # Pattern for append to end: '$a\'
    append_pattern = r"sed\s+-i\s+['\"]\$a\\"

    if (
        re.search(replace_pattern, command)
        or re.search(delete_pattern, command)
        or re.search(insert_pattern, command)
        or re.search(append_pattern, command)
    ):
        return (True, "valid_edit_command")

    return (False, "invalid_format")


# ----------------------------
# Eval logic
# ----------------------------
async def evaluate_single_sample(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    test_case: Dict[str, Any],
    sample: Dict[str, Any],
    sample_idx: int,
    args: Args,
    system_prompt: str,
    prompt_template: str,
    include_context: bool,
) -> List[Dict[str, Any]]:
    """
    Evaluate a single sample with concurrency control and retries.
    Returns a list of evaluation results (one per judge sample).
    """
    async with sem:
        delay = 0.25
        results = []
        expected_command = test_case["expected_command"]

        # Check if generated command is exact match with expected command
        if sample.get("exact_match", 0) == 1:
            print(f"Exact match found for task {test_case['task_id']}")
            return [
                {
                    "sample_idx": sample_idx,
                    "generated_command": sample["generated_command"],
                    "equivalent": 1,
                    "exact_match": sample["exact_match"],
                    "generated_command_empty": 0,
                    "format_valid": True,
                    "format_reason": "same_as_expected",
                }
            ]

        # Handle empty generated command
        if sample["generated_command"] == "":
            return [
                {
                    "sample_idx": sample_idx,
                    "choice_idx": None,
                    "task_id": test_case["task_id"],
                    "error": f"Empty generated command for task {test_case['task_id']}",
                    "equivalent": 0,
                    "generated_command_empty": 1,
                    "format_valid": False,
                    "format_reason": "empty_generated_command",
                }
            ]

        # Check command format validity (only applies strict checks if expected is a sed edit)
        format_valid, format_reason = check_command_format(
            sample["generated_command"], expected_command
        )

        # If format check failed, skip the judge and mark as non-equivalent
        if not format_valid:
            return [
                {
                    "sample_idx": sample_idx,
                    "choice_idx": None,
                    "generated_command": sample["generated_command"],
                    "equivalent": 0,
                    "exact_match": sample.get("exact_match", 0),
                    "generated_command_empty": 0,
                    "format_valid": format_valid,
                    "format_reason": format_reason,
                }
            ]

        for attempt in range(args.max_attempts):
            try:
                format_dict = {
                    "expected": expected_command,
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
                            "generated_command": sample["generated_command"],
                            "thinking_trace": thinking_trace,
                            "evaluation_results": result,
                            "equivalent": equivalent,
                            "exact_match": sample["exact_match"],
                            "generated_command_empty": 0,
                            "format_valid": format_valid,
                            "format_reason": format_reason,
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
                            "generated_command_empty": 0,
                            "format_valid": format_valid,
                            "format_reason": format_reason,
                        }
                    ]
                await asyncio.sleep(delay)
                delay *= 2

        return results


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
    Handles evaluation of all samples for a test case with parallelized sample evaluation.
    """
    if test_case.get("error", None) is not None:
        print(
            f"Returning failure object for task {test_case['task_id']} due to error: {test_case['error']}"
        )
        return {
            "task_id": test_case["task_id"],
            "error": test_case["error"],
            "judge_avg_at_n": 0.0,
            "judge_pass_at_n": 0,
            "num_generated_command_empty": 0,
            "empty_command_rate": 0.0,
            "num_format_correct": 0,
            "format_correct_rate": 0.0,
            "had_error": True,
        }

    samples = test_case.get("samples", [])
    if not samples:
        print(f"Returning failure object for task {test_case['task_id']} due to no samples")
        return {
            "task_id": test_case["task_id"],
            "error": "No samples",
            "judge_avg_at_n": 0.0,
            "judge_pass_at_n": 0,
            "num_generated_command_empty": 0,
            "empty_command_rate": 0.0,
            "num_format_correct": 0,
            "format_correct_rate": 0.0,
            "had_error": True,
        }

    # Evaluate all samples in parallel
    sample_tasks = [
        evaluate_single_sample(
            client,
            sem,
            test_case,
            sample,
            sample_idx,
            args,
            system_prompt,
            prompt_template,
            include_context,
        )
        for sample_idx, sample in enumerate(samples)
    ]
    sample_results_nested = await asyncio.gather(*sample_tasks)

    # Flatten results
    sample_results = []
    for results in sample_results_nested:
        sample_results.extend(results)

    # Compute metrics
    num_generation_samples = len(samples)
    expected_command = test_case["expected_command"]

    # Count format correct and empty commands from results
    # We need to count unique samples (by sample_idx) for format_correct and empty counts
    sample_format_valid = {}
    sample_empty = {}
    for r in sample_results:
        idx = r.get("sample_idx")
        if idx is not None:
            # Track format validity per sample (take first occurrence)
            if idx not in sample_format_valid:
                sample_format_valid[idx] = r.get("format_valid", False)
            # Track empty status per sample
            if idx not in sample_empty:
                sample_empty[idx] = r.get("generated_command_empty", 0) == 1

    num_format_correct = sum(1 for v in sample_format_valid.values() if v)
    num_generated_command_empty = sum(1 for v in sample_empty.values() if v)

    # Compute avg@n and pass@n
    num_judge_matches = sum(s.get("equivalent", 0) for s in sample_results)
    judge_avg_at_n = num_judge_matches / len(sample_results) if sample_results else 0.0
    judge_pass_at_n = int(num_judge_matches > 0)
    num_exact_matches = test_case.get("num_exact_matches", 0)

    # Compute empty command rate for this task
    empty_command_rate = (
        num_generated_command_empty / num_generation_samples if num_generation_samples else 0.0
    )

    # Compute format correct rate for this task
    format_correct_rate = (
        num_format_correct / num_generation_samples if num_generation_samples else 0.0
    )

    return {
        "task_id": test_case["task_id"],
        "context": test_case["context"],
        "expected_command": expected_command,
        "sample_evaluations": sample_results,
        "num_generation_samples": num_generation_samples,
        "num_judge_samples_per_generation": args.num_samples,
        "num_total_evaluations": len(sample_results),
        "num_judge_matches": num_judge_matches,
        "judge_avg_at_n": judge_avg_at_n,
        "judge_pass_at_n": judge_pass_at_n,
        "num_exact_matches": num_exact_matches,
        "num_generated_command_empty": num_generated_command_empty,
        "empty_command_rate": empty_command_rate,
        "num_format_correct": num_format_correct,
        "format_correct_rate": format_correct_rate,
        "had_error": any("error" in s for s in sample_results),
    }


async def run_single_eval(
    args: Args,
    generations_file: str,
    evaluations_file: str,
    eval_step: int,
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    system_prompt: str,
    prompt_template: str,
    include_context: bool,
    logger: Optional[LocalLogger] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single generations file and write results to evaluations file.
    Uses shared client and semaphore for efficiency in batch mode.
    Returns the evaluation scores dictionary.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {generations_file}")
    print(f"Output: {evaluations_file}")
    print(f"Step: {eval_step}")
    print(f"{'='*60}")

    loaded_data = load_dataset(generations_file)
    test_cases = loaded_data["generation_results"]

    config_generations = loaded_data["config_generations"]
    config_evaluations = args.__dict__
    metadata = {
        "config_generations": config_generations,
        "config_evaluations": config_evaluations,
    }

    # Initialize logger (local or wandb)
    logger = None
    if args.use_local_logger:
        run_id = args.wandb_id or args.wandb_name
        logger = LocalLogger(
            log_dir=args.local_log_dir,
            run_id=run_id,
            run_name=args.wandb_name,
            project=args.wandb_project,
            config=metadata,
            tags=args.wandb_tags,
        )
    else:
        wandb_init_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_name,
            "tags": args.wandb_tags,
            "group": args.wandb_group,
            "config": metadata,
        }

        if args.wandb_id:
            wandb_init_kwargs.update(
                {
                    "id": args.wandb_id,
                    "resume": "allow",
                }
            )
        wandb.init(**wandb_init_kwargs)

    if args.limit > 0:
        test_cases = test_cases[: args.limit]

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
    if os.path.exists(evaluations_file):
        os.remove(evaluations_file)

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

    os.makedirs(os.path.dirname(evaluations_file), exist_ok=True)
    total_judge_avg_at_n = sum(r.get("judge_avg_at_n", 0) for r in results) / len(results)
    total_judge_pass_at_n = sum(r.get("judge_pass_at_n", 0) for r in results)
    num_errors = sum(1 for r in results if r.get("had_error", False))

    # Calculate empty command rate: average across all tasks
    total_empty_command_rate = sum(r.get("empty_command_rate", 0) for r in results) / len(results)
    total_generated_command_empty = sum(r.get("num_generated_command_empty", 0) for r in results)

    # Calculate format correct rate: average across all tasks
    total_format_correct_rate = sum(r.get("format_correct_rate", 0) for r in results) / len(results)
    total_format_correct = sum(r.get("num_format_correct", 0) for r in results)
    total_format_incorrect = sum(
        r.get("num_generation_samples", 0) - r.get("num_format_correct", 0) for r in results
    )

    total_exact_match_avg_at_n = loaded_data["generation_scores"]["total_exact_match_avg_at_n"]
    total_exact_match_pass_at_n = loaded_data["generation_scores"]["total_exact_match_pass_at_n"]

    # Prepare metrics to log
    metrics_to_log = {
        "eval_step": eval_step,
        f"{args.wandb_eval_type}/total_test_cases": len(test_cases),
        f"{args.wandb_eval_type}/num_samples_per_task": loaded_data["config_generations"][
            "num_samples"
        ],
        f"{args.wandb_eval_type}/num_judge_samples_per_rollout": args.num_samples,
        f"{args.wandb_eval_type}/total_judge_avg_at_n": total_judge_avg_at_n,
        f"{args.wandb_eval_type}/total_judge_pass_at_n": total_judge_pass_at_n,
        f"{args.wandb_eval_type}/total_exact_match_avg_at_n": total_exact_match_avg_at_n,
        f"{args.wandb_eval_type}/total_exact_match_pass_at_n": total_exact_match_pass_at_n,
        f"{args.wandb_eval_type}/total_empty_command_rate": total_empty_command_rate,
        f"{args.wandb_eval_type}/total_generated_command_empty": total_generated_command_empty,
        f"{args.wandb_eval_type}/total_format_correct_rate": total_format_correct_rate,
        f"{args.wandb_eval_type}/total_format_correct": total_format_correct,
        f"{args.wandb_eval_type}/total_format_incorrect": total_format_incorrect,
        f"{args.wandb_eval_type}/num_errors": num_errors,
    }

    # Log metrics using appropriate logger
    if args.use_local_logger and logger is not None:
        logger.log(metrics_to_log)
    else:
        wandb.log(metrics_to_log)

    with open(evaluations_file, "w") as f:
        json.dump(
            {
                "metadata": metadata,
                "evaluation_scores": {
                    "total_test_cases": len(test_cases),
                    "num_samples_per_task": loaded_data["config_generations"]["num_samples"],
                    "num_judge_samples_per_rollout": args.num_samples,
                    "total_judge_avg_at_n": total_judge_avg_at_n,
                    "total_judge_pass_at_n": total_judge_pass_at_n,
                    "total_exact_match_avg_at_n": total_exact_match_avg_at_n,
                    "total_exact_match_pass_at_n": total_exact_match_pass_at_n,
                    "total_empty_command_rate": total_empty_command_rate,
                    "total_generated_command_empty": total_generated_command_empty,
                    "total_format_correct_rate": total_format_correct_rate,
                    "total_format_correct": total_format_correct,
                    "total_format_incorrect": total_format_incorrect,
                    "max_attempts": args.max_attempts,
                    "num_errors": num_errors,
                },
                "generation_results": results,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 50)
    print(f"--- Evaluation Complete (step {eval_step}) ---")
    print("=" * 50)
    print(f"Total Test Cases: {len(test_cases)}")
    print(f"Total Errors: {num_errors}")
    print(f"Total Judge Pass At N: {total_judge_pass_at_n}")
    print(f"Total Judge Avg At N: {total_judge_avg_at_n * 100:.2f}%")
    print(f"Total Exact Match Pass At N: {total_exact_match_pass_at_n}")
    print(f"Total Exact Match Avg At N: {total_exact_match_avg_at_n * 100:.2f}%")
    print(f"Empty Command Rate: {total_empty_command_rate * 100:.2f}%")
    print(f"Total Empty Commands: {total_generated_command_empty}")
    print(f"Format Correct Rate: {total_format_correct_rate * 100:.2f}%")
    print(f"Total Format Correct: {total_format_correct}")
    print(f"Total Format Incorrect: {total_format_incorrect}")
    print(f"Evaluations output file: {evaluations_file}")

    return {
        "eval_step": eval_step,
        "generations_file": generations_file,
        "evaluations_file": evaluations_file,
        "total_test_cases": len(test_cases),
        "total_judge_avg_at_n": total_judge_avg_at_n,
        "total_judge_pass_at_n": total_judge_pass_at_n,
        "total_exact_match_avg_at_n": total_exact_match_avg_at_n,
        "total_exact_match_pass_at_n": total_exact_match_pass_at_n,
        "total_empty_command_rate": total_empty_command_rate,
        "total_format_correct_rate": total_format_correct_rate,
    }


async def run_batch_eval(args: Args, base_url: str):
    """
    Run evaluation on multiple generation files with a single model load.
    This avoids the overhead of loading/unloading the judge model for each checkpoint.
    """
    eval_jobs = args.get_eval_jobs()

    print(f"\n{'#'*60}")
    print(f"# BATCH EVALUATION MODE")
    print(f"# Processing {len(eval_jobs)} evaluation job(s)")
    print(f"{'#'*60}\n")

    for i, (gen_file, eval_file, step) in enumerate(eval_jobs):
        print(f"  [{i+1}/{len(eval_jobs)}] Step {step}: {gen_file}")
    print()

    # Load prompts once (shared across all evaluations)
    with open(args.system_prompt_file, "r") as f:
        system_prompt = f.read()

    judge_prompt_file = (
        args.judge_prompt_file_with_context if args.include_context else args.judge_prompt_file
    )

    with open(judge_prompt_file, "r") as f:
        prompt_template = f.read()

    # Initialize logger (local or wandb) - shared across all evaluations
    logger = None
    if args.use_local_logger:
        run_id = args.wandb_id or args.wandb_name
        logger = LocalLogger(
            log_dir=args.local_log_dir,
            run_id=run_id,
            run_name=args.wandb_name,
            project=args.wandb_project,
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
            wandb_init_kwargs.update(
                {
                    "id": args.wandb_id,
                    "resume": "allow",
                    "dir": wandb_dir,
                }
            )
        wandb.init(**wandb_init_kwargs)

    # Reuse a single HTTP/2 client with a large pool (shared across all evaluations)
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

    # Process each evaluation job sequentially
    all_results = []
    for i, (gen_file, eval_file, step) in enumerate(eval_jobs):
        print(f"\n[{i+1}/{len(eval_jobs)}] Processing step {step}...")

        result = await run_single_eval(
            args=args,
            generations_file=gen_file,
            evaluations_file=eval_file,
            eval_step=step,
            client=client,
            sem=sem,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
            include_context=args.include_context,
            logger=logger,
        )
        all_results.append(result)

    await http.aclose()

    # Finish logging
    if args.use_local_logger:
        logger.finish()
    else:
        wandb.finish()

    # Print summary
    print("\n" + "#" * 60)
    print("# BATCH EVALUATION SUMMARY")
    print("#" * 60)
    for r in all_results:
        print(
            f"  Step {r['eval_step']:>5}: Judge Avg@N = {r['total_judge_avg_at_n']*100:5.2f}%, "
            f"Pass@N = {r['total_judge_pass_at_n']}, "
            f"Exact Avg@N = {r['total_exact_match_avg_at_n']*100:5.2f}%, "
            f"Empty Rate = {r['total_empty_command_rate']*100:5.2f}%, "
            f"Format Correct = {r['total_format_correct_rate']*100:5.2f}%"
        )
    print("#" * 60)


# ----------------------------
# Server launch + waiting
# ----------------------------
async def wait_for_server(base_url: str, timeout: float = 600.0) -> None:
    """
    Poll the server's OpenAI-compatible endpoint until it responds or timeout.
    We'll try a lightweight call to /models.
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

        await run_batch_eval(args, base_url=base_url)

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
