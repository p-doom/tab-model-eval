#!/usr/bin/env python3
"""
Convert HumanEval-MultiLineInfilling to YAML eval format.

Creates three versions of each task:
1. Direct completion - model sees file with gap, should fill it
2. Error recovery - model tries to run, gets error, views file, then fixes
3. Continuation - partial solution exists, model needs to finish

Usage:
    python src/input_pipeline/humaneval_infilling_to_testcases.py \
        --input_file /path/to/HumanEval-MultiLineInfilling.jsonl \
        --output_dir data/eval/humaneval_infilling \
        --versions direct,error  # Skip continuation
"""

import json
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import tyro
import yaml
from tqdm import tqdm


@dataclass
class Args:
    input_file: str = "HumanEval-MultiLineInfilling.jsonl"
    output_dir: str = "data/eval/humaneval_infilling"
    base_path: str = "data/eval/humaneval_infilling/tasks"
    # Which versions to generate: "direct", "error", "continuation", or "all"
    versions: str = "all"
    # Random seed for reproducibility
    seed: int = 42
    # Limit number of tasks (for testing), -1 for all
    limit: int = -1


def parse_humaneval_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a HumanEval-Infilling task and extract components."""
    prompt = task.get("prompt", "")
    suffix = task.get("suffix", "")
    canonical_solution = task.get("canonical_solution", "")

    # Split into lines
    prompt_lines = prompt.split("\n")
    suffix_lines = suffix.split("\n") if suffix else []
    solution_lines = canonical_solution.rstrip("\n").split("\n")

    # Remove trailing empty line from prompt if present
    while prompt_lines and prompt_lines[-1] == "":
        prompt_lines.pop()

    # Remove leading empty line from suffix if present
    while suffix_lines and suffix_lines[0] == "":
        suffix_lines.pop(0)

    return {
        "task_id": task.get("task_id", ""),
        "entry_point": task.get("entry_point", ""),
        "prompt": prompt,
        "suffix": suffix,
        "canonical_solution": canonical_solution,
        "test": task.get("test", ""),
        "prompt_lines": prompt_lines,
        "suffix_lines": suffix_lines,
        "solution_lines": solution_lines,
    }


def create_synthetic_file(
    prompt_lines: List[str],
    suffix_lines: List[str],
    gap_placeholder: str = "",
) -> Tuple[List[str], int]:
    """
    Create synthetic file content with a gap.
    Returns (file_lines, gap_line_number).
    """
    file_lines = prompt_lines.copy()

    # Add empty line(s) for the gap
    gap_line = len(file_lines) + 1  # 1-indexed
    file_lines.append(gap_placeholder)

    # Add suffix
    file_lines.extend(suffix_lines)

    return file_lines, gap_line


def create_direct_completion_task(
    parsed: Dict[str, Any],
    base_path: str,
) -> Dict[str, Any]:
    """
    Version 1: Direct completion
    Model sees ls, then cat of file with gap, should fill it directly.
    """
    entry_point = parsed["entry_point"]
    file_path = f"{base_path}/{entry_point}.py"

    # Create synthetic file with gap
    file_lines, gap_line = create_synthetic_file(
        parsed["prompt_lines"],
        parsed["suffix_lines"],
        gap_placeholder="",  # Empty line as gap
    )

    return {
        "task_id": f"humaneval_direct/{parsed['task_id']}",
        "description": "HumanEval infilling: direct completion",
        "humaneval_meta": {
            "original_task_id": parsed["task_id"],
            "entry_point": entry_point,
            "test": parsed["test"],
            "canonical_solution": parsed["canonical_solution"],
        },
        "states": [
            {
                "step": 0,
                "eval": "NO_EVAL",
                "terminal": {
                    "command": f"ls {base_path}/",
                    "output": f"{entry_point}.py",
                    "exit_code": 0,
                    "cwd": "/project",
                },
                "files": {},
                "cursor": None,
            },
            {
                "step": 1,
                "eval": "NO_EVAL",
                "terminal": None,
                "files": {file_path: "\n".join(file_lines)},
                "cursor": {"file": file_path, "line": gap_line, "column": 0},
            },
            {
                "step": 2,
                "eval": "EVAL",
                "judge_assertions": "Fill the missing lines in the editable region.",
                "terminal": None,
                "files": {
                    file_path: "\n".join(
                        parsed["prompt_lines"]
                        + parsed["canonical_solution"].rstrip("\n").split("\n")
                        + parsed["suffix_lines"]
                    )
                },
                "cursor": {"file": file_path, "line": gap_line, "column": 0},
            },
        ],
    }


def create_error_recovery_task(
    parsed: Dict[str, Any],
    base_path: str,
) -> Dict[str, Any]:
    """
    Version 2: Error recovery
    Model tries to run/import, gets IndentationError, views file, then fixes.
    """
    entry_point = parsed["entry_point"]
    file_path = f"{base_path}/{entry_point}.py"

    # Create synthetic file with gap
    file_lines, gap_line = create_synthetic_file(
        parsed["prompt_lines"],
        parsed["suffix_lines"],
        gap_placeholder="",
    )

    # The error line is where suffix starts (indented code without context)
    error_line = gap_line + 1 if parsed["suffix_lines"] else gap_line

    return {
        "task_id": f"humaneval_error/{parsed['task_id']}",
        "description": "HumanEval infilling: error recovery",
        "humaneval_meta": {
            "original_task_id": parsed["task_id"],
            "entry_point": entry_point,
            "test": parsed["test"],
            "canonical_solution": parsed["canonical_solution"],
        },
        "states": [
            {
                "step": 0,
                "eval": "NO_EVAL",
                "terminal": {
                    "command": f'python -c "from {entry_point} import {entry_point}"',
                    "output": f"  File \"{file_path}\", line {error_line}\n    {parsed['suffix_lines'][0] if parsed['suffix_lines'] else ''}\nIndentationError: unexpected indent",
                    "exit_code": 1,
                    "cwd": "/project",
                },
                "files": {},
                "cursor": None,
            },
            {
                "step": 1,
                "eval": "NO_EVAL",
                "terminal": None,
                "files": {file_path: "\n".join(file_lines)},
                "cursor": {"file": file_path, "line": gap_line, "column": 0},
            },
            {
                "step": 2,
                "eval": "EVAL",
                "judge_assertions": "Fill the missing lines in the editable region.",
                "terminal": None,
                "files": {
                    file_path: "\n".join(
                        parsed["prompt_lines"]
                        + parsed["canonical_solution"].rstrip("\n").split("\n")
                        + parsed["suffix_lines"]
                    )
                },
                "cursor": {"file": file_path, "line": gap_line, "column": 0},
            },
        ],
    }


def create_continuation_task(
    parsed: Dict[str, Any],
    base_path: str,
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    """
    Version 3: Continuation
    Model already started writing, file shows partial solution (cut mid-line),
    model needs to complete it.
    """
    entry_point = parsed["entry_point"]
    file_path = f"{base_path}/{entry_point}.py"

    solution = parsed["canonical_solution"]
    solution_lines = parsed["solution_lines"]

    # Need at least some content to cut
    if len(solution) < 10:
        return None

    # Cut at a random point (between 30-70% of first line, or after first line)
    first_line = solution_lines[0]
    if len(first_line) < 10:
        return None

    # Cut the first line at a random point
    cut_point = rng.randint(int(len(first_line) * 0.3), int(len(first_line) * 0.7))
    partial_first_line = first_line[:cut_point]

    # The partial solution is just the cut first line
    partial_solution = partial_first_line

    # Create file with partial solution
    file_lines = parsed["prompt_lines"].copy()
    partial_line_num = len(file_lines) + 1
    file_lines.append(partial_solution)  # Incomplete line
    file_lines.extend(parsed["suffix_lines"])

    # Create original file lines (before partial edit) for initial cat
    original_file_lines = parsed["prompt_lines"].copy()
    original_file_lines.append("")  # Empty gap
    original_file_lines.extend(parsed["suffix_lines"])

    return {
        "task_id": f"humaneval_continuation/{parsed['task_id']}",
        "description": "HumanEval infilling: continuation",
        "humaneval_meta": {
            "original_task_id": parsed["task_id"],
            "entry_point": entry_point,
            "test": parsed["test"],
            "canonical_solution": parsed["canonical_solution"],
            "partial_solution": partial_solution,
        },
        "states": [
            {
                "step": 0,
                "eval": "NO_EVAL",
                "terminal": {
                    "command": f"ls {base_path}/",
                    "output": f"{entry_point}.py",
                    "exit_code": 0,
                    "cwd": "/project",
                },
                "files": {},
                "cursor": None,
            },
            {
                "step": 1,
                "eval": "NO_EVAL",
                "terminal": None,
                "files": {file_path: "\n".join(original_file_lines)},
                "cursor": {"file": file_path, "line": partial_line_num, "column": 0},
            },
            {
                "step": 2,
                "eval": "NO_EVAL",
                "terminal": None,
                "files": {file_path: "\n".join(file_lines)},
                "cursor": {
                    "file": file_path,
                    "line": partial_line_num,
                    "column": len(partial_solution),
                },
            },
            {
                "step": 3,
                "eval": "EVAL",
                "judge_assertions": "Complete the partial line in the editable region.",
                "terminal": None,
                "files": {
                    file_path: "\n".join(
                        parsed["prompt_lines"]
                        + parsed["canonical_solution"].rstrip("\n").split("\n")
                        + parsed["suffix_lines"]
                    )
                },
                "cursor": {"file": file_path, "line": partial_line_num, "column": 0},
            },
        ],
    }


def convert_humaneval_to_tasks(args: Args) -> List[Dict[str, Any]]:
    """Main conversion function."""
    rng = random.Random(args.seed)

    # Load HumanEval tasks
    tasks = []
    with open(args.input_file, "r") as f:
        for line in f:
            tasks.append(json.loads(line))

    if args.limit > 0:
        tasks = tasks[: args.limit]

    print(f"Loaded {len(tasks)} HumanEval-Infilling tasks")

    test_cases = []
    versions_to_create = (
        args.versions.split(",") if args.versions != "all" else ["direct", "error", "continuation"]
    )

    for task in tqdm(tasks):
        parsed = parse_humaneval_task(task)

        # Skip tasks with empty canonical solution
        if not parsed["canonical_solution"].strip():
            continue

        if "direct" in versions_to_create:
            tc = create_direct_completion_task(parsed, args.base_path)
            test_cases.append(tc)

        if "error" in versions_to_create:
            tc = create_error_recovery_task(parsed, args.base_path)
            test_cases.append(tc)

        if "continuation" in versions_to_create:
            tc = create_continuation_task(parsed, args.base_path, rng)
            if tc:
                test_cases.append(tc)

    return test_cases


def write_yaml_tasks(output_dir: str, tasks: List[Dict[str, Any]]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for task in tqdm(tasks):
        task_id = task["task_id"].replace("/", "__")
        path = os.path.join(output_dir, f"{task_id}.yaml")
        with open(path, "w") as f:
            yaml.dump(task, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def main():
    args = tyro.cli(Args)
    tasks = convert_humaneval_to_tasks(args)
    write_yaml_tasks(args.output_dir, tasks)

    print(f"\nCreated {len(tasks)} tasks")
    print(f"Output directory: {args.output_dir}")

    version_counts = {}
    for tc in tqdm(tasks):
        version = tc["task_id"].split("/")[0]
        version_counts[version] = version_counts.get(version, 0) + 1

    print("\nBreakdown by version:")
    for version, count in sorted(version_counts.items()):
        print(f"  {version}: {count}")


if __name__ == "__main__":
    main()
