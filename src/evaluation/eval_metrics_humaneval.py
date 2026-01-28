import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import tyro
import yaml

from .eval_utils import (
    create_humaneval_sandbox,
    execute_sed_in_sandbox,
    is_humaneval_task,
    run_humaneval_test,
    cleanup_sandbox,
    validate_sed_command_for_sandbox,
)
from .yaml_output import load_generation_yaml


@dataclass
class Args:
    generations_file: str = "data/eval/generations/generations.yaml"
    metrics_file: str = "data/eval/metrics/humaneval_metrics.yaml"
    eval_step: int = 0
    limit: int = -1

    yaml_input_dir: str = ""
    sandbox_base_dir: Optional[str] = None
    sandbox_timeout: float = 10.0


def load_task_meta(yaml_dir: str) -> Dict[str, Dict[str, Any]]:
    tasks: Dict[str, Dict[str, Any]] = {}
    if not yaml_dir:
        return tasks
    for filename in sorted(os.listdir(yaml_dir)):
        if not filename.endswith(".yaml"):
            continue
        path = os.path.join(yaml_dir, filename)
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if data is None:
            continue
        task_id = data.get("task_id")
        if task_id:
            tasks[task_id] = data
    return tasks


def get_pre_eval_files(states: List[Dict[str, Any]]) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for i, state in enumerate(states):
        if state.get("eval") == "EVAL":
            if i > 0:
                prev = states[i - 1].get("files", {})
                if prev:
                    return prev
            break
        state_files = state.get("files", {})
        if state_files:
            files.update(state_files)
    return files


def run_humaneval_sample(
    generated_command: str,
    entry_point: str,
    test_code: str,
    file_path: str,
    file_content: str,
    sandbox_base_dir: Optional[str],
    sandbox_timeout: float,
) -> Dict[str, Any]:
    result = {
        "execution_success": False,
        "test_passed": False,
        "error": "",
        "sandbox_validated": False,
    }

    if not file_content:
        result["error"] = "empty_file_content"
        return result

    sandbox_dir = create_humaneval_sandbox(
        file_path=file_path,
        file_content=file_content,
        base_sandbox_dir=sandbox_base_dir,
    )

    try:
        allowed_paths = ["/home/user/projects/", entry_point]
        is_safe, reason, rewritten_cmd = validate_sed_command_for_sandbox(
            generated_command,
            sandbox_dir,
            allowed_paths,
        )
        if not is_safe or rewritten_cmd is None:
            result["error"] = f"sandbox_validation_failed_{reason}"
            return result

        result["sandbox_validated"] = True

        success, _, stderr = execute_sed_in_sandbox(rewritten_cmd, sandbox_dir, sandbox_timeout)
        if not success:
            result["error"] = f"sed_execution_failed_{stderr}"
            return result

        result["execution_success"] = True

        if not test_code:
            result["error"] = "missing_test_code"
            return result

        passed, test_error = run_humaneval_test(sandbox_dir, entry_point, test_code)
        result["test_passed"] = passed
        if not passed:
            result["error"] = f"test_failed_{test_error}"

        return result
    finally:
        cleanup_sandbox(sandbox_dir)


def evaluate_task(
    task_result: Dict[str, Any],
    task_meta: Optional[Dict[str, Any]],
    sandbox_base_dir: Optional[str],
    sandbox_timeout: float,
) -> Dict[str, Any]:
    task_id = task_result.get("task_id", "unknown")
    states = task_result.get("states", [])
    samples = task_result.get("samples", [])

    humaneval_meta = (task_meta or {}).get("humaneval_meta", {})
    entry_point = humaneval_meta.get("entry_point", "")
    test_code = humaneval_meta.get("test", "")

    pre_eval_files = get_pre_eval_files(states)
    if len(pre_eval_files) != 1:
        return {
            "task_id": task_id,
            "error": "expected_single_file_state",
            "num_samples": len(samples),
            "sample_results": [],
            "pass_at_1": 0,
        }

    if not entry_point or not test_code:
        return {
            "task_id": task_id,
            "error": "missing_humaneval_meta",
            "num_samples": len(samples),
            "sample_results": [],
            "pass_at_1": 0,
        }

    file_path, file_content = next(iter(pre_eval_files.items()))

    sample_results = []
    for sample in samples:
        sample_result = run_humaneval_sample(
            generated_command=sample.get("predicted_raw", ""),
            entry_point=entry_point,
            test_code=test_code,
            file_path=file_path,
            file_content=file_content,
            sandbox_base_dir=sandbox_base_dir,
            sandbox_timeout=sandbox_timeout,
        )
        sample_result["sample_idx"] = sample.get("sample_idx", 0)
        sample_results.append(sample_result)

    num_samples = len(sample_results)
    num_test_passed = sum(1 for s in sample_results if s["test_passed"])
    num_exec_success = sum(1 for s in sample_results if s["execution_success"])
    num_validated = sum(1 for s in sample_results if s["sandbox_validated"])

    return {
        "task_id": task_id,
        "num_samples": num_samples,
        "num_execution_success": num_exec_success,
        "num_test_passed": num_test_passed,
        "num_sandbox_validated": num_validated,
        "execution_rate": num_exec_success / num_samples if num_samples else 0.0,
        "pass_rate": num_test_passed / num_samples if num_samples else 0.0,
        "pass_at_1": int(num_test_passed > 0),
        "sample_results": sample_results,
    }


def run_metrics_eval(
    generations_file: str,
    metrics_file: str,
    eval_step: int,
    limit: int,
    yaml_input_dir: str,
    sandbox_base_dir: Optional[str],
    sandbox_timeout: float,
) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"Evaluating: {generations_file}")
    print(f"Output: {metrics_file}")
    print(f"{'='*60}")

    data = load_generation_yaml(generations_file)
    results = data.get("results", [])
    config = data.get("config", {})

    if limit > 0:
        results = results[:limit]

    if not yaml_input_dir:
        yaml_input_dir = config.get("yaml_input_dir", "")

    task_meta = load_task_meta(yaml_input_dir) if yaml_input_dir else {}

    humaneval_results = [r for r in results if is_humaneval_task(r.get("task_id", ""))]
    print(f"Evaluating {len(humaneval_results)} HumanEval tasks...")

    task_metrics = [
        evaluate_task(r, task_meta.get(r.get("task_id", "")), sandbox_base_dir, sandbox_timeout)
        for r in humaneval_results
    ]

    num_tasks = len(task_metrics)
    total_samples = sum(t["num_samples"] for t in task_metrics)
    total_exec = sum(t["num_execution_success"] for t in task_metrics)
    total_pass = sum(t["num_test_passed"] for t in task_metrics)
    total_pass_at_1 = sum(t["pass_at_1"] for t in task_metrics)

    scores = {
        "total_tasks": num_tasks,
        "total_samples": total_samples,
        "total_execution_success": total_exec,
        "total_test_passed": total_pass,
        "total_pass_at_1": total_pass_at_1,
        "execution_rate": total_exec / total_samples if total_samples else 0.0,
        "pass_rate": total_pass / total_samples if total_samples else 0.0,
        "pass_at_1_rate": total_pass_at_1 / num_tasks if num_tasks else 0.0,
    }

    output = {
        "metadata": {
            "generations_file": generations_file,
            "eval_step": eval_step,
            "yaml_input_dir": yaml_input_dir,
            "config": config,
        },
        "metrics_scores": scores,
        "task_metrics": task_metrics,
    }

    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, "w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print("\n" + "=" * 50)
    print(f"HumanEval Sandbox Metrics Complete (step {eval_step})")
    print("=" * 50)
    print(f"Tasks: {num_tasks}")
    print(f"Samples: {total_samples}")
    print(f"Execution Success: {total_exec}/{total_samples} ({scores['execution_rate']*100:.1f}%)")
    print(f"Pass@1: {total_pass_at_1}/{num_tasks} ({scores['pass_at_1_rate']*100:.1f}%)")
    print(f"Output: {metrics_file}")

    return scores


def main() -> None:
    args = tyro.cli(Args)
    run_metrics_eval(
        generations_file=args.generations_file,
        metrics_file=args.metrics_file,
        eval_step=args.eval_step,
        limit=args.limit,
        yaml_input_dir=args.yaml_input_dir,
        sandbox_base_dir=args.sandbox_base_dir,
        sandbox_timeout=args.sandbox_timeout,
    )


if __name__ == "__main__":
    main()
