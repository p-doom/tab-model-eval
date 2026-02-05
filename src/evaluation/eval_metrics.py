import difflib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import tyro
import yaml
import wandb

from .eval_utils import find_matching_path
from .yaml_output import load_generation_yaml


@dataclass
class Args:
    wandb_project: str = "tab-model-eval"
    wandb_name: str = "metrics_eval"
    wandb_eval_type: str = "metrics_eval"
    wandb_tags: list[str] = field(default_factory=list)
    wandb_id: Optional[str] = None
    wandb_group: str = "evals"

    generations_file: str = "data/eval/generations/generations.yaml"
    metrics_file: str = "data/eval/metrics/metrics.yaml"
    eval_step: int = 0
    limit: int = -1

    generations_files: str = ""
    metrics_files: str = ""
    eval_steps: str = ""

    def get_eval_jobs(self) -> List[tuple[str, str, int]]:
        if self.generations_files and self.metrics_files and self.eval_steps:
            gen = [f.strip() for f in self.generations_files.split(",") if f.strip()]
            met = [f.strip() for f in self.metrics_files.split(",") if f.strip()]
            steps = [int(s.strip()) for s in self.eval_steps.split(",") if s.strip()]
            if len(gen) == len(met) == len(steps):
                return list(zip(gen, met, steps))
            raise ValueError("Batch lists must be same length")
        return [(self.generations_file, self.metrics_file, self.eval_step)]


def compute_file_similarity(expected: str, predicted: str) -> float:
    if expected == predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0
    return difflib.SequenceMatcher(None, expected, predicted).ratio()


def compute_line_diff_metrics(expected: str, predicted: str) -> Dict[str, Any]:
    expected_lines = expected.split("\n")
    predicted_lines = predicted.split("\n")
    diff = list(difflib.unified_diff(expected_lines, predicted_lines, lineterm=""))

    additions = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

    return {
        "num_expected_lines": len(expected_lines),
        "num_predicted_lines": len(predicted_lines),
        "num_additions": additions,
        "num_deletions": deletions,
        "total_diff_lines": additions + deletions,
    }


def _normalize_terminal_command(command: str) -> str:
    text = (command or "").strip()
    if not text:
        return ""

    match = re.search(r"```(?:bash)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _cursor_exact_match(
    expected_cursor: Optional[Dict[str, Any]],
    predicted_cursor: Optional[Dict[str, Any]],
) -> Optional[bool]:
    if expected_cursor is None:
        return None
    if predicted_cursor is None:
        return False

    expected_file = expected_cursor.get("file")
    predicted_file = predicted_cursor.get("file")

    if expected_file:
        matched = find_matching_path([expected_file], predicted_file)
        if matched is None:
            return False

    return predicted_cursor.get("line") == expected_cursor.get("line") and predicted_cursor.get(
        "column"
    ) == expected_cursor.get("column")


def evaluate_sample(
    expected_files: Dict[str, str],
    predicted_files: Optional[Dict[str, str]],
    predicted_raw: str,
    prediction_error: Optional[str],
    expected_cursor: Optional[Dict[str, Any]] = None,
    predicted_cursor: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "has_prediction": predicted_files is not None,
        "prediction_error": prediction_error,
        "is_empty_prediction": not predicted_raw.strip(),
        "cursor_exact_match": _cursor_exact_match(expected_cursor, predicted_cursor),
    }

    if prediction_error or predicted_files is None:
        cursor_match = result.get("cursor_exact_match")
        sample_pass = False if cursor_match is not None else False
        result.update(
            {
                "file_exact_match": False,
                "avg_similarity": 0.0,
                "files_evaluated": 0,
                "sample_pass": sample_pass,
            }
        )
        return result

    file_matches = []
    similarities = []
    diff_metrics = []

    for path, expected_content in expected_files.items():
        predicted_content = predicted_files.get(path, "")
        exact_match = expected_content == predicted_content
        similarity = compute_file_similarity(expected_content, predicted_content)
        diff = compute_line_diff_metrics(expected_content, predicted_content)

        file_matches.append(exact_match)
        similarities.append(similarity)
        diff_metrics.append(
            {"path": path, "exact_match": exact_match, "similarity": similarity, **diff}
        )

    unexpected_files = set(predicted_files.keys()) - set(expected_files.keys())

    result.update(
        {
            "file_exact_match": all(file_matches) if file_matches else False,
            "avg_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
            "files_evaluated": len(file_matches),
            "num_files_matched": sum(file_matches),
            "num_unexpected_files": len(unexpected_files),
            "file_details": diff_metrics,
        }
    )

    cursor_match = result.get("cursor_exact_match")
    result["sample_pass"] = (
        result["file_exact_match"] and cursor_match
        if cursor_match is not None
        else result["file_exact_match"]
    )

    return result


def evaluate_terminal_sample(
    expected_command: str,
    predicted_raw: str,
    prediction_error: Optional[str],
) -> Dict[str, Any]:
    expected_norm = _normalize_terminal_command(expected_command)
    predicted_norm = _normalize_terminal_command(predicted_raw)
    terminal_exact_match = bool(predicted_norm) and expected_norm == predicted_norm

    return {
        "has_prediction": bool(predicted_norm),
        "prediction_error": prediction_error,
        "is_empty_prediction": not predicted_norm,
        "terminal_exact_match": terminal_exact_match,
        "sample_pass": terminal_exact_match,
    }


def evaluate_task(task_result: Dict[str, Any]) -> Dict[str, Any]:
    task_id = task_result.get("task_id", "unknown")
    states = task_result.get("states", [])
    samples = task_result.get("samples", [])

    expected_files = {}
    expected_cursor = None
    expected_terminal_command = ""
    assertions = None
    for state in states:
        eval = state.get("eval", "NO_EVAL")
        if eval == "EVAL":
            expected_files = state.get("files", {})
            expected_cursor = state.get("cursor")
            terminal = state.get("terminal") or {}
            expected_terminal_command = terminal.get("command", "") if terminal else ""
            assertions = state.get("judge_assertions")
            break
    eval_mode = "terminal" if expected_terminal_command else "state"

    sample_results = []
    for sample in samples:
        if eval_mode == "terminal":
            sample_eval = evaluate_terminal_sample(
                expected_command=expected_terminal_command,
                predicted_raw=sample.get("predicted_raw", ""),
                prediction_error=sample.get("prediction_error"),
            )
        else:
            sample_eval = evaluate_sample(
                expected_files=expected_files,
                predicted_files=sample.get("predicted_files"),
                predicted_raw=sample.get("predicted_raw", ""),
                prediction_error=sample.get("prediction_error"),
                expected_cursor=expected_cursor,
                predicted_cursor=sample.get("predicted_cursor"),
            )
        sample_eval["sample_idx"] = sample.get("sample_idx", 0)
        sample_eval["exact_match_raw"] = sample.get("exact_match", 0)
        sample_results.append(sample_eval)

    num_samples = len(sample_results)
    num_has_prediction = sum(1 for s in sample_results if s["has_prediction"])
    num_empty = sum(1 for s in sample_results if s["is_empty_prediction"])
    num_pass = sum(1 for s in sample_results if s.get("sample_pass", False))

    if num_samples > 0:
        if eval_mode == "terminal":
            num_file_exact_match = 0
            file_exact_match_rate = 0.0
            avg_similarity = 0.0
            num_cursor_evaluated = 0
            num_cursor_exact_match = 0
            cursor_exact_match_rate = 0.0
            num_terminal_exact_match = sum(
                1 for s in sample_results if s.get("terminal_exact_match", False)
            )
            terminal_exact_match_rate = num_terminal_exact_match / num_samples
        else:
            num_file_exact_match = sum(1 for s in sample_results if s["file_exact_match"])
            file_exact_match_rate = num_file_exact_match / num_samples
            avg_similarity = sum(s["avg_similarity"] for s in sample_results) / num_samples
            cursor_results = [
                s["cursor_exact_match"]
                for s in sample_results
                if s.get("cursor_exact_match") is not None
            ]
            num_cursor_evaluated = len(cursor_results)
            num_cursor_exact_match = sum(1 for c in cursor_results if c is True)
            cursor_exact_match_rate = (
                num_cursor_exact_match / num_cursor_evaluated if num_cursor_evaluated > 0 else 0.0
            )
            num_terminal_exact_match = 0
            terminal_exact_match_rate = 0.0
    else:
        num_file_exact_match = 0
        file_exact_match_rate = 0.0
        avg_similarity = 0.0
        num_cursor_evaluated = 0
        num_cursor_exact_match = 0
        cursor_exact_match_rate = 0.0
        num_terminal_exact_match = 0
        terminal_exact_match_rate = 0.0

    return {
        "task_id": task_id,
        "assertions": assertions,
        "eval_mode": eval_mode,
        "num_samples": num_samples,
        "num_pass": num_pass,
        "pass_rate": num_pass / num_samples if num_samples > 0 else 0.0,
        "num_file_exact_match": num_file_exact_match,
        "num_cursor_evaluated": num_cursor_evaluated,
        "num_cursor_exact_match": num_cursor_exact_match,
        "num_terminal_exact_match": num_terminal_exact_match,
        "num_has_prediction": num_has_prediction,
        "num_empty_predictions": num_empty,
        "file_exact_match_rate": file_exact_match_rate,
        "cursor_exact_match_rate": cursor_exact_match_rate,
        "terminal_exact_match_rate": terminal_exact_match_rate,
        "avg_similarity": avg_similarity,
        "pass_at_1": int(num_pass > 0),
        "sample_results": sample_results,
    }


def run_metrics_eval(
    generations_file: str,
    metrics_file: str,
    eval_step: int,
    limit: int = -1,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
    wandb_eval_type: str = "metrics_eval",
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

    print(f"Evaluating {len(results)} tasks...")

    task_metrics = [evaluate_task(r) for r in results]

    num_tasks = len(task_metrics)
    total_samples = sum(t["num_samples"] for t in task_metrics)
    total_file_exact = sum(t["num_file_exact_match"] for t in task_metrics)
    total_cursor_exact = sum(t["num_cursor_exact_match"] for t in task_metrics)
    total_terminal_exact = sum(t["num_terminal_exact_match"] for t in task_metrics)
    total_pass = sum(t["num_pass"] for t in task_metrics)
    total_pass_at_1 = sum(t["pass_at_1"] for t in task_metrics)
    total_empty = sum(t["num_empty_predictions"] for t in task_metrics)
    num_state_tasks = sum(1 for t in task_metrics if t.get("eval_mode") == "state")
    num_terminal_tasks = sum(1 for t in task_metrics if t.get("eval_mode") == "terminal")

    avg_file_exact_rate = (
        sum(t["file_exact_match_rate"] for t in task_metrics if t.get("eval_mode") == "state")
        / num_state_tasks
        if num_state_tasks > 0
        else 0.0
    )
    avg_similarity = (
        sum(t["avg_similarity"] for t in task_metrics if t.get("eval_mode") == "state")
        / num_state_tasks
        if num_state_tasks > 0
        else 0.0
    )
    avg_terminal_exact_rate = (
        sum(
            t["terminal_exact_match_rate"] for t in task_metrics if t.get("eval_mode") == "terminal"
        )
        / num_terminal_tasks
        if num_terminal_tasks > 0
        else 0.0
    )

    scores = {
        "total_tasks": num_tasks,
        "num_state_tasks": num_state_tasks,
        "num_terminal_tasks": num_terminal_tasks,
        "total_samples": total_samples,
        "total_pass": total_pass,
        "total_file_exact_match": total_file_exact,
        "total_cursor_exact_match": total_cursor_exact,
        "total_terminal_exact_match": total_terminal_exact,
        "total_pass_at_1": total_pass_at_1,
        "total_empty_predictions": total_empty,
        "avg_file_exact_match_rate": avg_file_exact_rate,
        "avg_terminal_exact_match_rate": avg_terminal_exact_rate,
        "avg_similarity": avg_similarity,
        "pass_at_1_rate": total_pass_at_1 / num_tasks if num_tasks > 0 else 0.0,
        "pass_rate": total_pass / total_samples if total_samples > 0 else 0.0,
        "empty_rate": total_empty / total_samples if total_samples > 0 else 0.0,
    }

    output = {
        "metadata": {
            "generations_file": generations_file,
            "eval_step": eval_step,
            "config": config,
        },
        "metrics_scores": scores,
        "task_metrics": task_metrics,
    }

    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, "w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print("\n" + "=" * 50)
    print(f"Metrics Evaluation Complete (step {eval_step})")
    print("=" * 50)
    print(f"Tasks: {num_tasks}")
    print(f"Samples: {total_samples}")
    if num_state_tasks > 0:
        print(
            f"File Exact Match: {total_file_exact}/{total_samples} ({avg_file_exact_rate*100:.1f}%)"
        )
    if num_terminal_tasks > 0:
        print(
            f"Terminal Exact Match: {total_terminal_exact}/{total_samples} ({avg_terminal_exact_rate*100:.1f}%)"
        )
    print(f"Pass@1: {total_pass_at_1}/{num_tasks} ({scores['pass_at_1_rate']*100:.1f}%)")
    print(f"Avg Similarity: {avg_similarity*100:.1f}%")
    print(f"Output: {metrics_file}")

    if wandb_run is not None:
        wandb_run.log(
            {
                "eval_step": eval_step,
                f"{wandb_eval_type}/total_tasks": num_tasks,
                f"{wandb_eval_type}/num_state_tasks": num_state_tasks,
                f"{wandb_eval_type}/num_terminal_tasks": num_terminal_tasks,
                f"{wandb_eval_type}/total_samples": total_samples,
                f"{wandb_eval_type}/total_pass": total_pass,
                f"{wandb_eval_type}/total_file_exact_match": total_file_exact,
                f"{wandb_eval_type}/total_cursor_exact_match": total_cursor_exact,
                f"{wandb_eval_type}/total_terminal_exact_match": total_terminal_exact,
                f"{wandb_eval_type}/total_pass_at_1": total_pass_at_1,
                f"{wandb_eval_type}/avg_file_exact_match_rate": avg_file_exact_rate,
                f"{wandb_eval_type}/avg_terminal_exact_match_rate": avg_terminal_exact_rate,
                f"{wandb_eval_type}/avg_similarity": avg_similarity,
                f"{wandb_eval_type}/pass_at_1_rate": scores["pass_at_1_rate"],
                f"{wandb_eval_type}/pass_rate": scores["pass_rate"],
                f"{wandb_eval_type}/empty_rate": scores["empty_rate"],
            }
        )

    return scores


def main():
    args = tyro.cli(Args)
    jobs = args.get_eval_jobs()
    print(f"Running {len(jobs)} evaluation job(s)")

    wandb_run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        id=args.wandb_id,
        resume="allow" if args.wandb_id else None,
        group=args.wandb_group,
        tags=args.wandb_tags,
        config={"eval_type": args.wandb_eval_type},
    )

    all_results = []
    for gen_file, met_file, step in jobs:
        result = run_metrics_eval(
            gen_file,
            met_file,
            step,
            args.limit,
            wandb_run=wandb_run,
            wandb_eval_type=args.wandb_eval_type,
        )
        all_results.append(result)

    if len(all_results) > 1:
        print("\n" + "#" * 60)
        print("BATCH SUMMARY")
        print("#" * 60)
        for job, result in zip(jobs, all_results):
            print(
                f"  Step {job[2]}: Pass@1={result['pass_at_1_rate']*100:.1f}%, Similarity={result['avg_similarity']*100:.1f}%"
            )

    if wandb_run is not None:
        wandb_run.finish()

    print("\nDone")


if __name__ == "__main__":
    main()
