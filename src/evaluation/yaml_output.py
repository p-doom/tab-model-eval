import os
from typing import Any, Dict, List, Optional

import yaml


def build_state(
    step: int,
    eval_tag: str,
    files: Optional[Dict[str, str]] = None,
    cursor: Optional[Dict[str, Any]] = None,
    terminal: Optional[Dict[str, Any]] = None,
    judge_assertions: Optional[str] = None,
    predicted_files: Optional[Dict[str, str]] = None,
    predicted_raw: Optional[str] = None,
    prediction_error: Optional[str] = None,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {"step": step, "eval": eval_tag}
    if files is not None:
        state["files"] = files
    if cursor is not None:
        state["cursor"] = cursor
    if terminal is not None:
        state["terminal"] = terminal
    if judge_assertions is not None:
        state["judge_assertions"] = judge_assertions
    if predicted_files is not None:
        state["predicted_files"] = predicted_files
    if predicted_raw is not None:
        state["predicted_raw"] = predicted_raw
    if prediction_error is not None:
        state["prediction_error"] = prediction_error
    return state


def build_sample_prediction(
    sample_idx: int,
    step: int,
    predicted_files: Optional[Dict[str, str]],
    predicted_raw: str,
    exact_match: int,
    prediction_error: Optional[str] = None,
) -> Dict[str, Any]:
    pred: Dict[str, Any] = {
        "sample_idx": sample_idx,
        "step": step,
        "predicted_raw": predicted_raw,
        "exact_match": exact_match,
    }
    if predicted_files is not None:
        pred["predicted_files"] = predicted_files
    if prediction_error is not None:
        pred["prediction_error"] = prediction_error
    return pred


def build_generation_result(
    task_id: str,
    format_type: str,
    states: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "task_id": task_id,
        "format": format_type,
        "states": states,
        "samples": samples,
    }
    if description:
        result["description"] = description
    if config:
        result["config"] = config

    if samples:
        num_samples = len(samples)
        num_exact = sum(s.get("exact_match", 0) for s in samples)
        result["metrics"] = {
            "num_samples": num_samples,
            "num_exact_matches": num_exact,
            "exact_match_rate": num_exact / num_samples if num_samples > 0 else 0.0,
            "pass_at_1": int(num_exact > 0),
        }

    return result


def write_yaml_output(
    output_path: str,
    results: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_tasks = len(results)
    total_samples = sum(r.get("metrics", {}).get("num_samples", 0) for r in results)
    total_exact = sum(r.get("metrics", {}).get("num_exact_matches", 0) for r in results)
    total_pass = sum(r.get("metrics", {}).get("pass_at_1", 0) for r in results)

    output: Dict[str, Any] = {
        "generation_scores": {
            "total_tasks": total_tasks,
            "total_samples": total_samples,
            "total_exact_matches": total_exact,
            "exact_match_rate": total_exact / total_samples if total_samples > 0 else 0.0,
            "pass_at_1_rate": total_pass / total_tasks if total_tasks > 0 else 0.0,
        },
        "results": results,
    }
    if config:
        output["config"] = config

    with open(output_path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def write_per_task_yaml(output_dir: str, results: List[Dict[str, Any]]) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for result in results:
        task_id = result.get("task_id", "unknown")
        safe_id = task_id.replace("/", "_").replace("\\", "_")
        path = os.path.join(output_dir, f"{safe_id}.yaml")
        with open(path, "w") as f:
            yaml.dump(result, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        paths.append(path)
    return paths


def load_generation_yaml(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def get_predictions_from_yaml(yaml_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    predictions = []
    for result in yaml_data.get("results", []):
        task_id = result.get("task_id", "unknown")
        states = result.get("states", [])
        samples = result.get("samples", [])

        state_map = {s.get("step"): s for s in states}

        for sample in samples:
            step = sample.get("step", 0)
            state = state_map.get(step, {})
            predictions.append(
                {
                    "task_id": task_id,
                    "sample_idx": sample.get("sample_idx", 0),
                    "step": step,
                    "expected_files": state.get("files", {}),
                    "predicted_files": sample.get("predicted_files", {}),
                    "predicted_raw": sample.get("predicted_raw", ""),
                    "exact_match": sample.get("exact_match", 0),
                    "prediction_error": sample.get("prediction_error"),
                    "judge_assertions": state.get("judge_assertions"),
                }
            )
    return predictions
