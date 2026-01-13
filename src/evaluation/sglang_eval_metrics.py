"""
Fast, deterministic metrics evaluation for code completion models.

This script evaluates generated commands using deterministic metrics:
- Format compliance (valid sed patterns, no file redirection, etc.)
- Empty command detection
- Exact match (from generation file)
- HumanEval execution (sandbox-based code execution and testing)
- IFEval evaluation (instruction-following evaluation)

This is designed to run quickly without any LLM calls, enabling fast
iteration on format validation and cheap evaluation of many checkpoints.

For LLM-as-a-judge evaluation, use sglang_eval_judge.py which can consume
the output of this script.

Usage:
    # Standard evaluation
    python sglang_eval_metrics.py \
        --generations-file data/eval/output/generations.json \
        --metrics-file data/eval/output/metrics.json \
        --eval-step 1000

    # With HumanEval sandbox execution
    python sglang_eval_metrics.py \
        --generations-file data/eval/output/humaneval_generations.json \
        --metrics-file data/eval/output/humaneval_metrics.json \
        --eval-step 1000 \
        --run-humaneval-sandbox

    # With IFEval evaluation
    python sglang_eval_metrics.py \
        --generations-file data/eval/output/ifeval_generations.json \
        --metrics-file data/eval/output/ifeval_metrics.json \
        --eval-step 1000 \
        --run-ifeval \
        --ifeval-testcases-file data/eval/ifeval/ifeval_testcases.jsonl
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional

import tyro
import wandb
from tqdm import tqdm

from sglang_eval_utils import (
    LocalLogger,
    check_command_format,
    evaluate_humaneval_sample,
    evaluate_ifeval_sample,
    is_humaneval_task,
    is_ifeval_task,
    load_dataset,
    save_dataset,
)


def load_humaneval_testcases(filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    Load HumanEval test cases from JSONL file and index by task_id.
    Returns a dict mapping task_id -> test case (with humaneval_meta).
    """
    testcases = {}
    with open(filepath, "r") as f:
        for line in f:
            tc = json.loads(line)
            task_id = tc.get("task_id", "")
            testcases[task_id] = tc
    return testcases


def load_ifeval_testcases(filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    Load IFEval test cases from JSONL file and index by task_id.
    Returns a dict mapping task_id -> test case (with ifeval_meta).
    """
    testcases = {}
    with open(filepath, "r") as f:
        for line in f:
            tc = json.loads(line)
            task_id = tc.get("task_id", "")
            testcases[task_id] = tc
    return testcases


# ----------------------------
# Argument definitions
# ----------------------------
@dataclass
class Args:
    # Wandb logging
    wandb_project: str = "llm-coding-agent"
    wandb_name: str = "validation_set_metrics"
    wandb_eval_type: str = "next_action_validation_set"
    wandb_tags: list[str] = field(default_factory=lambda: ["val_mini", "metrics_eval"])
    wandb_id: str | None = None
    wandb_group: str = "debug"

    # Single-file mode
    generations_file: str = ""
    metrics_file: str = ""
    eval_step: int = 0

    # Batch mode: comma-separated lists
    generations_files: str = ""
    metrics_files: str = ""
    eval_steps: str = ""

    limit: int = -1

    # Local logging for offline mode
    use_local_logger: bool = False
    local_log_dir: str = "data/eval/local_logs"

    # HumanEval sandbox execution
    run_humaneval_sandbox: bool = False
    sandbox_base_dir: Optional[str] = None  # If None, uses system temp
    sandbox_timeout: float = 10.0  # Timeout for test execution
    # Path to HumanEval test cases file (JSONL with humaneval_meta)
    # Required for sandbox execution to get test code
    humaneval_testcases_file: str = ""
    # Number of parallel workers for sandbox execution
    num_workers: int = 32

    # IFEval evaluation
    run_ifeval: bool = False
    # Path to IFEval test cases file (JSONL with ifeval_meta)
    # Required for IFEval execution to get instruction metadata
    ifeval_testcases_file: str = ""

    def get_eval_jobs(self) -> List[tuple[str, str, int]]:
        """
        Returns a list of (generations_file, metrics_file, eval_step) tuples.
        """
        if self.generations_files and self.metrics_files and self.eval_steps:
            gen_files = [f.strip() for f in self.generations_files.split(",") if f.strip()]
            metrics_files = [f.strip() for f in self.metrics_files.split(",") if f.strip()]
            steps = [int(s.strip()) for s in self.eval_steps.split(",") if s.strip()]

            if not (len(gen_files) == len(metrics_files) == len(steps)):
                raise ValueError(
                    f"Batch mode requires equal-length lists for generations_files ({len(gen_files)}), "
                    f"metrics_files ({len(metrics_files)}), and eval_steps ({len(steps)})"
                )

            return list(zip(gen_files, metrics_files, steps))
        elif self.generations_file and self.metrics_file:
            return [(self.generations_file, self.metrics_file, self.eval_step)]
        else:
            raise ValueError(
                "Either provide single-file args (generations_file, metrics_file) "
                "or batch args (generations_files, metrics_files, eval_steps)"
            )


# ----------------------------
# Metrics evaluation logic
# ----------------------------
def evaluate_sample_metrics(
    sample: Dict[str, Any],
    sample_idx: int,
    expected_command: str,
    context: Optional[List[Dict[str, Any]]] = None,
    humaneval_meta: Optional[Dict[str, Any]] = None,
    run_humaneval_sandbox: bool = False,
    sandbox_base_dir: Optional[str] = None,
    ifeval_meta: Optional[Dict[str, Any]] = None,
    run_ifeval: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a single sample using deterministic metrics.
    Returns metrics dict including format validity, exact match, and optionally HumanEval/IFEval results.

    Note: IFEval uses the full response_text (raw model output), not the extracted generated_command.
    """
    generated_command = sample.get("generated_command", "")
    # For IFEval, use the full response text, not just the extracted command
    response_text = sample.get("response_text", "")

    # Check for empty command (for format/HumanEval evaluation)
    generated_command_empty = generated_command == ""

    # Check format validity (only meaningful if there's a generated command)
    if generated_command_empty:
        format_valid = False
        format_reason = "empty_generated_command"
    else:
        format_valid, format_reason = check_command_format(generated_command, expected_command)

    result = {
        "sample_idx": sample_idx,
        "generated_command": generated_command,
        "generated_command_empty": generated_command_empty,
        "format_valid": format_valid,
        "format_reason": format_reason,
        "exact_match": sample.get("exact_match", 0),
    }

    # Run HumanEval sandbox evaluation if enabled and metadata is available
    # Skip if format is invalid - no point running tests on malformed commands
    if run_humaneval_sandbox and humaneval_meta and context:
        if not format_valid:
            # Skip sandbox execution for format-invalid commands
            result.update(
                {
                    "humaneval_execution_success": False,
                    "humaneval_test_passed": False,
                    "humaneval_error": f"skipped_format_invalid_{format_reason}",
                    "humaneval_sandbox_validated": False,
                }
            )
        else:
            humaneval_result = evaluate_humaneval_sample(
                generated_command=generated_command,
                context=context,
                humaneval_meta=humaneval_meta,
                base_sandbox_dir=sandbox_base_dir,
            )
            result.update(
                {
                    "humaneval_execution_success": humaneval_result["execution_success"],
                    "humaneval_test_passed": humaneval_result["test_passed"],
                    "humaneval_error": humaneval_result["error"],
                    "humaneval_sandbox_validated": humaneval_result["sandbox_validated"],
                }
            )
            if result["humaneval_error"] and "test_failed" not in result["humaneval_error"]:
                print(f"HumanEval error: {result['humaneval_error']}")

    # Run IFEval evaluation if enabled and metadata is available
    # IFEval uses the full response_text, not the extracted bash command
    if run_ifeval and ifeval_meta:
        ifeval_result = evaluate_ifeval_sample(
            response=response_text,
            ifeval_meta=ifeval_meta,
        )
        result.update(
            {
                "ifeval_strict_pass": ifeval_result["ifeval_strict_pass"],
                "ifeval_loose_pass": ifeval_result["ifeval_loose_pass"],
                "ifeval_strict_instructions_followed": ifeval_result[
                    "ifeval_strict_instructions_followed"
                ],
                "ifeval_loose_instructions_followed": ifeval_result[
                    "ifeval_loose_instructions_followed"
                ],
                "ifeval_total_instructions": ifeval_result["ifeval_total_instructions"],
                "ifeval_error": ifeval_result["ifeval_error"],
            }
        )

    return result


def evaluate_task_metrics(
    test_case: Dict[str, Any],
    run_humaneval_sandbox: bool = False,
    sandbox_base_dir: Optional[str] = None,
    humaneval_testcases: Optional[Dict[str, Dict[str, Any]]] = None,
    run_ifeval: bool = False,
    ifeval_testcases: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Evaluate all samples for a test case using deterministic metrics.

    Args:
        test_case: Generation result with samples
        run_humaneval_sandbox: Whether to run HumanEval sandbox execution
        sandbox_base_dir: Base directory for sandboxes
        humaneval_testcases: Dict mapping task_id -> test case (with humaneval_meta).
                            Required for HumanEval sandbox execution to get test code.
        run_ifeval: Whether to run IFEval evaluation
        ifeval_testcases: Dict mapping task_id -> test case (with ifeval_meta).
                         Required for IFEval evaluation to get instruction metadata.
    """
    task_id = test_case.get("task_id", "")
    is_humaneval = is_humaneval_task(task_id)
    is_ifeval = is_ifeval_task(task_id)

    # Base error result
    def make_error_result(error: str) -> Dict[str, Any]:
        result = {
            "task_id": task_id,
            "error": error,
            "had_error": True,
            "num_samples": 0,
            "num_format_correct": 0,
            "num_generated_command_empty": 0,
            "num_exact_matches": 0,
            "format_correct_rate": 0.0,
            "empty_command_rate": 0.0,
            "exact_match_rate": 0.0,
            "sample_metrics": [],
        }
        if run_humaneval_sandbox and is_humaneval:
            result.update(
                {
                    "num_humaneval_execution_success": 0,
                    "num_humaneval_test_passed": 0,
                    "humaneval_execution_rate": 0.0,
                    "humaneval_pass_rate": 0.0,
                }
            )
        if run_ifeval and is_ifeval:
            result.update(
                {
                    "num_ifeval_strict_pass": 0,
                    "num_ifeval_loose_pass": 0,
                    "total_ifeval_strict_instructions_followed": 0,
                    "total_ifeval_loose_instructions_followed": 0,
                    "total_ifeval_instructions": 0,
                    "ifeval_strict_pass_rate": 0.0,
                    "ifeval_loose_pass_rate": 0.0,
                }
            )
        return result

    # Handle error cases from generation
    if test_case.get("error", None) is not None:
        return make_error_result(test_case["error"])

    samples = test_case.get("samples", [])
    if not samples:
        return make_error_result("No samples")

    expected_command = test_case.get("expected_command", "")
    context = test_case.get("context", [])

    # Get humaneval_meta from test case or external testcases file
    humaneval_meta = test_case.get("humaneval_meta", None)
    if humaneval_meta is None and humaneval_testcases and task_id in humaneval_testcases:
        # Get metadata from external testcases file
        tc_data = humaneval_testcases[task_id]
        humaneval_meta = tc_data.get("humaneval_meta", None)
        # Also get context from testcases if not in generation result
        if not context:
            context = tc_data.get("context", [])

    # Get ifeval_meta from test case or external testcases file
    ifeval_meta = test_case.get("ifeval_meta", None)
    if ifeval_meta is None and ifeval_testcases and task_id in ifeval_testcases:
        # Get metadata from external testcases file
        tc_data = ifeval_testcases[task_id]
        ifeval_meta = tc_data.get("ifeval_meta", None)

    # Evaluate each sample
    sample_metrics = [
        evaluate_sample_metrics(
            sample=sample,
            sample_idx=idx,
            expected_command=expected_command,
            context=context,
            humaneval_meta=humaneval_meta,
            run_humaneval_sandbox=run_humaneval_sandbox and is_humaneval,
            sandbox_base_dir=sandbox_base_dir,
            ifeval_meta=ifeval_meta,
            run_ifeval=run_ifeval and is_ifeval,
        )
        for idx, sample in enumerate(samples)
    ]

    # Aggregate metrics
    num_samples = len(samples)
    num_format_correct = sum(1 for m in sample_metrics if m["format_valid"])
    num_empty = sum(1 for m in sample_metrics if m["generated_command_empty"])
    num_exact_matches = sum(m["exact_match"] for m in sample_metrics)

    result = {
        "task_id": task_id,
        "context": context,
        "expected_command": expected_command,
        "had_error": False,
        "num_samples": num_samples,
        "num_format_correct": num_format_correct,
        "num_generated_command_empty": num_empty,
        "num_exact_matches": num_exact_matches,
        "format_correct_rate": num_format_correct / num_samples if num_samples else 0.0,
        "empty_command_rate": num_empty / num_samples if num_samples else 0.0,
        "exact_match_rate": num_exact_matches / num_samples if num_samples else 0.0,
        "sample_metrics": sample_metrics,
        # Preserve original samples for downstream judge evaluation
        "samples": samples,
    }

    # Add HumanEval metrics if applicable
    if run_humaneval_sandbox and is_humaneval:
        num_exec_success = sum(
            1 for m in sample_metrics if m.get("humaneval_execution_success", False)
        )
        num_test_passed = sum(1 for m in sample_metrics if m.get("humaneval_test_passed", False))
        result.update(
            {
                "num_humaneval_execution_success": num_exec_success,
                "num_humaneval_test_passed": num_test_passed,
                "humaneval_execution_rate": num_exec_success / num_samples if num_samples else 0.0,
                "humaneval_pass_rate": num_test_passed / num_samples if num_samples else 0.0,
                "humaneval_pass_at_1": int(num_test_passed > 0),
            }
        )

    # Add IFEval metrics if applicable
    if run_ifeval and is_ifeval:
        num_strict_pass = sum(1 for m in sample_metrics if m.get("ifeval_strict_pass", False))
        num_loose_pass = sum(1 for m in sample_metrics if m.get("ifeval_loose_pass", False))
        total_strict_followed = sum(
            m.get("ifeval_strict_instructions_followed", 0) for m in sample_metrics
        )
        total_loose_followed = sum(
            m.get("ifeval_loose_instructions_followed", 0) for m in sample_metrics
        )
        total_instructions = sum(m.get("ifeval_total_instructions", 0) for m in sample_metrics)
        result.update(
            {
                "num_ifeval_strict_pass": num_strict_pass,
                "num_ifeval_loose_pass": num_loose_pass,
                "total_ifeval_strict_instructions_followed": total_strict_followed,
                "total_ifeval_loose_instructions_followed": total_loose_followed,
                "total_ifeval_instructions": total_instructions,
                "ifeval_strict_pass_rate": num_strict_pass / num_samples if num_samples else 0.0,
                "ifeval_loose_pass_rate": num_loose_pass / num_samples if num_samples else 0.0,
                "ifeval_strict_pass_at_1": int(num_strict_pass > 0),
                "ifeval_loose_pass_at_1": int(num_loose_pass > 0),
            }
        )

    return result


def run_single_metrics_eval(
    args: Args,
    generations_file: str,
    metrics_file: str,
    eval_step: int,
    logger=None,
    humaneval_testcases: Optional[Dict[str, Dict[str, Any]]] = None,
    ifeval_testcases: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Run metrics evaluation on a single generations file.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating metrics: {generations_file}")
    print(f"Output: {metrics_file}")
    print(f"Step: {eval_step}")
    if args.run_humaneval_sandbox:
        print(f"HumanEval sandbox execution: ENABLED")
        if humaneval_testcases:
            print(f"HumanEval testcases loaded: {len(humaneval_testcases)} tasks")
    if args.run_ifeval:
        print(f"IFEval evaluation: ENABLED")
        if ifeval_testcases:
            print(f"IFEval testcases loaded: {len(ifeval_testcases)} tasks")
    print(f"{'='*60}")

    loaded_data = load_dataset(generations_file)
    test_cases = loaded_data["generation_results"]
    config_generations = loaded_data["config_generations"]

    if args.limit > 0:
        test_cases = test_cases[: args.limit]

    # Check if we have HumanEval and IFEval tasks
    num_humaneval_tasks = sum(1 for tc in test_cases if is_humaneval_task(tc.get("task_id", "")))
    num_ifeval_tasks = sum(1 for tc in test_cases if is_ifeval_task(tc.get("task_id", "")))
    print(
        f"Processing {len(test_cases)} test cases ({num_humaneval_tasks} HumanEval, {num_ifeval_tasks} IFEval tasks)..."
    )

    # Evaluate all tasks - use parallel processing if HumanEval sandbox is enabled
    if args.run_humaneval_sandbox and args.num_workers > 1:
        print(f"Using {args.num_workers} parallel workers for sandbox execution...")

        # Create a partial function with fixed arguments
        eval_func = partial(
            evaluate_task_metrics,
            run_humaneval_sandbox=args.run_humaneval_sandbox,
            sandbox_base_dir=args.sandbox_base_dir,
            humaneval_testcases=humaneval_testcases,
            run_ifeval=args.run_ifeval,
            ifeval_testcases=ifeval_testcases,
        )

        results = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(eval_func, tc): tc for tc in test_cases}

            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                results.append(future.result())
    else:
        # Sequential evaluation
        results = [
            evaluate_task_metrics(
                tc,
                run_humaneval_sandbox=args.run_humaneval_sandbox,
                sandbox_base_dir=args.sandbox_base_dir,
                humaneval_testcases=humaneval_testcases,
                run_ifeval=args.run_ifeval,
                ifeval_testcases=ifeval_testcases,
            )
            for tc in tqdm(test_cases, desc="Evaluating")
        ]

    # Sort by task_id
    results.sort(key=lambda x: x["task_id"])

    # Aggregate scores
    num_tasks = len(results)
    total_format_correct = sum(r["num_format_correct"] for r in results)
    total_format_incorrect = sum(r["num_samples"] - r["num_format_correct"] for r in results)
    total_empty = sum(r["num_generated_command_empty"] for r in results)
    total_exact_matches = sum(r["num_exact_matches"] for r in results)
    total_samples = sum(r["num_samples"] for r in results)
    num_errors = sum(1 for r in results if r.get("had_error", False))

    # Rates averaged over tasks
    avg_format_correct_rate = sum(r["format_correct_rate"] for r in results) / num_tasks
    avg_empty_command_rate = sum(r["empty_command_rate"] for r in results) / num_tasks
    avg_exact_match_rate = sum(r["exact_match_rate"] for r in results) / num_tasks

    # Also get the exact match metrics from generation file (for consistency)
    gen_exact_match_avg = loaded_data["generation_scores"].get("total_exact_match_avg_at_n", 0)
    gen_exact_match_pass = loaded_data["generation_scores"].get("total_exact_match_pass_at_n", 0)

    scores = {
        "total_test_cases": num_tasks,
        "total_samples": total_samples,
        "num_samples_per_task": config_generations.get("num_samples", 0),
        "total_format_correct": total_format_correct,
        "total_format_incorrect": total_format_incorrect,
        "total_generated_command_empty": total_empty,
        "total_exact_matches": total_exact_matches,
        "avg_format_correct_rate": avg_format_correct_rate,
        "avg_empty_command_rate": avg_empty_command_rate,
        "avg_exact_match_rate": avg_exact_match_rate,
        "gen_exact_match_avg_at_n": gen_exact_match_avg,
        "gen_exact_match_pass_at_n": gen_exact_match_pass,
        "num_errors": num_errors,
    }

    # HumanEval-specific metrics
    humaneval_results = [r for r in results if "humaneval_pass_rate" in r]
    if humaneval_results:
        total_humaneval_exec_success = sum(
            r["num_humaneval_execution_success"] for r in humaneval_results
        )
        total_humaneval_test_passed = sum(r["num_humaneval_test_passed"] for r in humaneval_results)
        total_humaneval_samples = sum(r["num_samples"] for r in humaneval_results)
        avg_humaneval_pass_rate = sum(r["humaneval_pass_rate"] for r in humaneval_results) / len(
            humaneval_results
        )
        total_humaneval_pass_at_1 = sum(r.get("humaneval_pass_at_1", 0) for r in humaneval_results)

        scores.update(
            {
                "num_humaneval_tasks": len(humaneval_results),
                "total_humaneval_execution_success": total_humaneval_exec_success,
                "total_humaneval_test_passed": total_humaneval_test_passed,
                "total_humaneval_samples": total_humaneval_samples,
                "avg_humaneval_pass_rate": avg_humaneval_pass_rate,
                "total_humaneval_pass_at_1": total_humaneval_pass_at_1,
                "humaneval_pass_at_1_rate": (
                    total_humaneval_pass_at_1 / len(humaneval_results) if humaneval_results else 0.0
                ),
            }
        )

    # IFEval-specific metrics
    ifeval_results = [r for r in results if "ifeval_strict_pass_rate" in r]
    if ifeval_results:
        total_ifeval_strict_pass = sum(r["num_ifeval_strict_pass"] for r in ifeval_results)
        total_ifeval_loose_pass = sum(r["num_ifeval_loose_pass"] for r in ifeval_results)
        total_ifeval_samples = sum(r["num_samples"] for r in ifeval_results)
        total_ifeval_strict_followed = sum(
            r["total_ifeval_strict_instructions_followed"] for r in ifeval_results
        )
        total_ifeval_loose_followed = sum(
            r["total_ifeval_loose_instructions_followed"] for r in ifeval_results
        )
        total_ifeval_instructions = sum(r["total_ifeval_instructions"] for r in ifeval_results)
        avg_ifeval_strict_pass_rate = sum(
            r["ifeval_strict_pass_rate"] for r in ifeval_results
        ) / len(ifeval_results)
        avg_ifeval_loose_pass_rate = sum(r["ifeval_loose_pass_rate"] for r in ifeval_results) / len(
            ifeval_results
        )
        total_ifeval_strict_pass_at_1 = sum(
            r.get("ifeval_strict_pass_at_1", 0) for r in ifeval_results
        )
        total_ifeval_loose_pass_at_1 = sum(
            r.get("ifeval_loose_pass_at_1", 0) for r in ifeval_results
        )

        scores.update(
            {
                "num_ifeval_tasks": len(ifeval_results),
                "total_ifeval_strict_pass": total_ifeval_strict_pass,
                "total_ifeval_loose_pass": total_ifeval_loose_pass,
                "total_ifeval_samples": total_ifeval_samples,
                "total_ifeval_strict_instructions_followed": total_ifeval_strict_followed,
                "total_ifeval_loose_instructions_followed": total_ifeval_loose_followed,
                "total_ifeval_instructions": total_ifeval_instructions,
                "avg_ifeval_strict_pass_rate": avg_ifeval_strict_pass_rate,
                "avg_ifeval_loose_pass_rate": avg_ifeval_loose_pass_rate,
                "total_ifeval_strict_pass_at_1": total_ifeval_strict_pass_at_1,
                "total_ifeval_loose_pass_at_1": total_ifeval_loose_pass_at_1,
                "ifeval_strict_pass_at_1_rate": (
                    total_ifeval_strict_pass_at_1 / len(ifeval_results) if ifeval_results else 0.0
                ),
                "ifeval_loose_pass_at_1_rate": (
                    total_ifeval_loose_pass_at_1 / len(ifeval_results) if ifeval_results else 0.0
                ),
                "ifeval_instruction_accuracy_strict": (
                    total_ifeval_strict_followed / total_ifeval_instructions
                    if total_ifeval_instructions
                    else 0.0
                ),
                "ifeval_instruction_accuracy_loose": (
                    total_ifeval_loose_followed / total_ifeval_instructions
                    if total_ifeval_instructions
                    else 0.0
                ),
            }
        )

    # Log metrics
    metrics_to_log = {
        "eval_step": eval_step,
        f"{args.wandb_eval_type}/total_test_cases": num_tasks,
        f"{args.wandb_eval_type}/total_samples": total_samples,
        f"{args.wandb_eval_type}/total_format_correct": total_format_correct,
        f"{args.wandb_eval_type}/total_format_incorrect": total_format_incorrect,
        f"{args.wandb_eval_type}/total_generated_command_empty": total_empty,
        f"{args.wandb_eval_type}/avg_format_correct_rate": avg_format_correct_rate,
        f"{args.wandb_eval_type}/avg_empty_command_rate": avg_empty_command_rate,
        f"{args.wandb_eval_type}/avg_exact_match_rate": avg_exact_match_rate,
        f"{args.wandb_eval_type}/gen_exact_match_avg_at_n": gen_exact_match_avg,
        f"{args.wandb_eval_type}/gen_exact_match_pass_at_n": gen_exact_match_pass,
        f"{args.wandb_eval_type}/num_errors": num_errors,
    }

    # Add HumanEval metrics to log
    if humaneval_results:
        metrics_to_log.update(
            {
                f"{args.wandb_eval_type}/num_humaneval_tasks": len(humaneval_results),
                f"{args.wandb_eval_type}/avg_humaneval_pass_rate": avg_humaneval_pass_rate,
                f"{args.wandb_eval_type}/total_humaneval_pass_at_1": total_humaneval_pass_at_1,
                f"{args.wandb_eval_type}/humaneval_pass_at_1_rate": scores[
                    "humaneval_pass_at_1_rate"
                ],
            }
        )

    # Add IFEval metrics to log
    if ifeval_results:
        metrics_to_log.update(
            {
                f"{args.wandb_eval_type}/num_ifeval_tasks": len(ifeval_results),
                f"{args.wandb_eval_type}/avg_ifeval_strict_pass_rate": avg_ifeval_strict_pass_rate,
                f"{args.wandb_eval_type}/avg_ifeval_loose_pass_rate": avg_ifeval_loose_pass_rate,
                f"{args.wandb_eval_type}/total_ifeval_strict_pass_at_1": total_ifeval_strict_pass_at_1,
                f"{args.wandb_eval_type}/total_ifeval_loose_pass_at_1": total_ifeval_loose_pass_at_1,
                f"{args.wandb_eval_type}/ifeval_strict_pass_at_1_rate": scores[
                    "ifeval_strict_pass_at_1_rate"
                ],
                f"{args.wandb_eval_type}/ifeval_loose_pass_at_1_rate": scores[
                    "ifeval_loose_pass_at_1_rate"
                ],
                f"{args.wandb_eval_type}/ifeval_instruction_accuracy_strict": scores[
                    "ifeval_instruction_accuracy_strict"
                ],
                f"{args.wandb_eval_type}/ifeval_instruction_accuracy_loose": scores[
                    "ifeval_instruction_accuracy_loose"
                ],
            }
        )

    if args.use_local_logger and logger is not None:
        logger.log(metrics_to_log)
    else:
        wandb.log(metrics_to_log)

    # Save output
    output_data = {
        "metadata": {
            "config_generations": config_generations,
            "config_metrics": {
                "run_humaneval_sandbox": args.run_humaneval_sandbox,
                "sandbox_timeout": args.sandbox_timeout,
                "run_ifeval": args.run_ifeval,
            },
            "eval_step": eval_step,
            "generations_file": generations_file,
        },
        "metrics_scores": scores,
        "metrics_results": results,
    }
    save_dataset(metrics_file, output_data)

    # Print summary
    print("\n" + "=" * 50)
    print(f"--- Metrics Evaluation Complete (step {eval_step}) ---")
    print("=" * 50)
    print(f"Total Test Cases: {num_tasks}")
    print(f"Total Samples: {total_samples}")
    print(f"Format Correct Rate: {avg_format_correct_rate * 100:.2f}%")
    print(f"Empty Command Rate: {avg_empty_command_rate * 100:.2f}%")
    print(f"Exact Match Rate: {avg_exact_match_rate * 100:.2f}%")
    print(f"Total Format Correct: {total_format_correct}")
    print(f"Total Format Incorrect: {total_format_incorrect}")
    print(f"Total Empty Commands: {total_empty}")
    print(f"Errors: {num_errors}")

    if humaneval_results:
        print(f"\n--- HumanEval Results ---")
        print(f"HumanEval Tasks: {len(humaneval_results)}")
        print(f"HumanEval Pass Rate (avg): {avg_humaneval_pass_rate * 100:.2f}%")
        print(
            f"HumanEval Pass@1: {total_humaneval_pass_at_1}/{len(humaneval_results)} ({scores['humaneval_pass_at_1_rate'] * 100:.2f}%)"
        )
        print(f"Total Tests Passed: {total_humaneval_test_passed}/{total_humaneval_samples}")

    if ifeval_results:
        print(f"\n--- IFEval Results ---")
        print(f"IFEval Tasks: {len(ifeval_results)}")
        print(f"IFEval Strict Pass Rate (avg): {avg_ifeval_strict_pass_rate * 100:.2f}%")
        print(f"IFEval Loose Pass Rate (avg): {avg_ifeval_loose_pass_rate * 100:.2f}%")
        print(
            f"IFEval Strict Pass@1: {total_ifeval_strict_pass_at_1}/{len(ifeval_results)} ({scores['ifeval_strict_pass_at_1_rate'] * 100:.2f}%)"
        )
        print(
            f"IFEval Loose Pass@1: {total_ifeval_loose_pass_at_1}/{len(ifeval_results)} ({scores['ifeval_loose_pass_at_1_rate'] * 100:.2f}%)"
        )
        print(
            f"Instruction Accuracy (strict): {scores['ifeval_instruction_accuracy_strict'] * 100:.2f}%"
        )
        print(
            f"Instruction Accuracy (loose): {scores['ifeval_instruction_accuracy_loose'] * 100:.2f}%"
        )

    print(f"\nOutput file: {metrics_file}")

    return {
        "eval_step": eval_step,
        "generations_file": generations_file,
        "metrics_file": metrics_file,
        **scores,
    }


def run_batch_metrics_eval(args: Args):
    """
    Run metrics evaluation on multiple generation files.
    """
    eval_jobs = args.get_eval_jobs()

    print(f"\n{'#'*60}")
    print(f"# BATCH METRICS EVALUATION")
    print(f"# Processing {len(eval_jobs)} evaluation job(s)")
    print(f"{'#'*60}\n")

    for i, (gen_file, metrics_file, step) in enumerate(eval_jobs):
        print(f"  [{i+1}/{len(eval_jobs)}] Step {step}: {gen_file}")
    print()

    # Load HumanEval testcases if sandbox execution is enabled
    humaneval_testcases = None
    if args.run_humaneval_sandbox and args.humaneval_testcases_file:
        print(f"Loading HumanEval testcases from: {args.humaneval_testcases_file}")
        humaneval_testcases = load_humaneval_testcases(args.humaneval_testcases_file)
        print(f"Loaded {len(humaneval_testcases)} HumanEval testcases")
    elif args.run_humaneval_sandbox and not args.humaneval_testcases_file:
        print("WARNING: HumanEval sandbox enabled but no testcases file provided.")
        print("         Use --humaneval-testcases-file to specify the testcases file.")

    # Load IFEval testcases if IFEval evaluation is enabled
    ifeval_testcases = None
    if args.run_ifeval and args.ifeval_testcases_file:
        print(f"Loading IFEval testcases from: {args.ifeval_testcases_file}")
        ifeval_testcases = load_ifeval_testcases(args.ifeval_testcases_file)
        print(f"Loaded {len(ifeval_testcases)} IFEval testcases")
    elif args.run_ifeval and not args.ifeval_testcases_file:
        print("WARNING: IFEval evaluation enabled but no testcases file provided.")
        print("         Use --ifeval-testcases-file to specify the testcases file.")

    # Initialize logger
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

    # Process each evaluation job
    all_results = []
    for i, (gen_file, metrics_file, step) in enumerate(eval_jobs):
        print(f"\n[{i+1}/{len(eval_jobs)}] Processing step {step}...")
        result = run_single_metrics_eval(
            args=args,
            generations_file=gen_file,
            metrics_file=metrics_file,
            eval_step=step,
            logger=logger,
            humaneval_testcases=humaneval_testcases,
            ifeval_testcases=ifeval_testcases,
        )
        all_results.append(result)

    # Finish logging
    if args.use_local_logger and logger is not None:
        logger.finish()
    else:
        wandb.finish()

    # Print summary
    print("\n" + "#" * 60)
    print("# BATCH METRICS EVALUATION SUMMARY")
    print("#" * 60)
    for r in all_results:
        line = (
            f"  Step {r['eval_step']:>5}: "
            f"Format Correct = {r['avg_format_correct_rate']*100:5.2f}%, "
            f"Empty Rate = {r['avg_empty_command_rate']*100:5.2f}%, "
            f"Exact Match = {r['avg_exact_match_rate']*100:5.2f}%"
        )
        if "avg_humaneval_pass_rate" in r:
            line += f", HumanEval Pass = {r['avg_humaneval_pass_rate']*100:5.2f}%"
        if "avg_ifeval_strict_pass_rate" in r:
            line += f", IFEval Strict = {r['avg_ifeval_strict_pass_rate']*100:5.2f}%"
        print(line)
    print("#" * 60)


def main():
    args = tyro.cli(Args)
    run_batch_metrics_eval(args)
    print("Done")


if __name__ == "__main__":
    main()
