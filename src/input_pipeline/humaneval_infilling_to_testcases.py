#!/usr/bin/env python3
"""
Convert HumanEval-MultiLineInfilling to eval test cases format.

Creates three versions of each task:
1. Direct completion - model sees file with gap, should fill it
2. Error recovery - model tries to run, gets error, views file, then fixes
3. Continuation - partial solution exists, model needs to finish

Usage:
    python humaneval_infilling_to_testcases.py \
        --input_file /path/to/HumanEval-MultiLineInfilling.jsonl \
        --output_file /path/to/humaneval_testcases.jsonl

    # Generate only specific versions
    python src/input_pipeline/humaneval_infilling_to_testcases.py \
        --input_file ../HumanEval-MultiLineInfilling.jsonl \
        --output_file data/eval/humaneval_infilling/humaneval_testcases.jsonl \
        --versions direct,error  # Skip continuation
"""

import json
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import tyro


@dataclass
class Args:
    input_file: str = "HumanEval-MultiLineInfilling.jsonl"
    output_file: str = "data/eval/humaneval_infilling/humaneval_testcases.jsonl"
    base_path: str = "data/eval/humaneval_infilling/tasks"
    # Which versions to generate: "direct", "error", "continuation", or "all"
    versions: str = "all"
    # Random seed for reproducibility
    seed: int = 42
    # Limit number of tasks (for testing), -1 for all
    limit: int = -1


def format_line_number(n: int) -> str:
    """Format line number as 6-char right-aligned + tab."""
    return f"{n:6d}\t"


def create_numbered_file_content(lines: List[str], start_line: int = 1) -> str:
    """Create file content with line numbers in the expected format."""
    result = []
    for i, line in enumerate(lines):
        result.append(f"{format_line_number(start_line + i)}{line}")
    return "\n".join(result)


def escape_for_sed(text: str) -> str:
    """
    Escape text for use in sed 'c\' command.
    - Backslashes need escaping
    - Single quotes need special handling
    - Ampersand is special in sed replacement
    """
    # Escape backslashes first
    text = text.replace("\\", "\\\\")
    # Escape ampersand (special in sed)
    text = text.replace("&", "\\&")
    # Handle single quotes by breaking out and escaping
    text = text.replace("'", "'\"'\"'")
    return text


def create_sed_command(
    file_path: str,
    start_line: int,
    end_line: int,
    new_content: str,
    viewport_start: int,
    viewport_end: int,
) -> str:
    r"""
    Create a sed command to replace lines, followed by cat to show viewport.
    
    Uses shell-style backslash line continuation to match training data format:
    sed -i 'START,ENDc\
    LINE1\
    LINE2' FILE && cat -n FILE | sed -n 'VSTART,VENDp'
    
    Each line except the last ends with \ for shell continuation.
    """
    lines = new_content.rstrip("\n").split("\n")

    if len(lines) == 1:
        # Single line replacement
        escaped_line = escape_for_sed(lines[0])
        sed_cmd = f"sed -i '{start_line},{end_line}c\\\n{escaped_line}' {file_path}"
    else:
        # Multi-line replacement with shell backslash continuation
        # Each line ends with \ except the last
        escaped_lines = [escape_for_sed(line) for line in lines]
        # Join with backslash + newline for shell continuation
        replacement = "\\\n".join(escaped_lines)
        sed_cmd = f"sed -i '{start_line},{end_line}c\\\n{replacement}' {file_path}"

    # Chain with viewport view
    full_cmd = f"{sed_cmd} && cat -n {file_path} | sed -n '{viewport_start},{viewport_end}p'"

    return f"```bash\n{full_cmd}\n```"


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


def create_direct_completion_testcase(
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

    # Calculate viewport for after edit (21 lines around gap)
    viewport_start = max(1, gap_line - 10)
    viewport_end = min(len(file_lines) + len(parsed["solution_lines"]) - 1, gap_line + 10)

    # Create context turns
    context = [
        # ls to explore
        {
            "role": "assistant",
            "content": f"```bash\nls {base_path}/\n```",
            "eval_tag": "NO_EVAL",
        },
        {
            "role": "user",
            "content": f"<stdout>\n{entry_point}.py\n</stdout>",
            "eval_tag": None,
        },
        # cat to view file
        {
            "role": "assistant",
            "content": f"```bash\ncat -n {file_path}\n```",
            "eval_tag": "NO_EVAL",
        },
        {
            "role": "user",
            "content": f"<stdout>\n{create_numbered_file_content(file_lines)}\n</stdout>",
            "eval_tag": None,
        },
    ]

    # Expected response: sed to fill the gap
    expected_response = create_sed_command(
        file_path=file_path,
        start_line=gap_line,
        end_line=gap_line,
        new_content=parsed["canonical_solution"],
        viewport_start=viewport_start,
        viewport_end=viewport_end,
    )

    return {
        "task_id": f"humaneval_direct/{parsed['task_id']}",
        "context": context,
        "expected_final_response": expected_response,
        "humaneval_meta": {
            "original_task_id": parsed["task_id"],
            "entry_point": entry_point,
            "test": parsed["test"],
            "canonical_solution": parsed["canonical_solution"],
        },
    }


def create_error_recovery_testcase(
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

    # Calculate viewport
    viewport_start = max(1, gap_line - 10)
    viewport_end = min(len(file_lines) + len(parsed["solution_lines"]) - 1, gap_line + 10)

    # Create context turns
    context = [
        # Try to import/run the module
        {
            "role": "assistant",
            "content": f'```bash\npython -c "from {entry_point} import {entry_point}"\n```',
            "eval_tag": "NO_EVAL",
        },
        {
            "role": "user",
            "content": f"<stdout>\n  File \"{file_path}\", line {error_line}\n    {parsed['suffix_lines'][0] if parsed['suffix_lines'] else ''}\nIndentationError: unexpected indent\n</stdout>",
            "eval_tag": None,
        },
        # View the file to understand the issue
        {
            "role": "assistant",
            "content": f"```bash\ncat -n {file_path}\n```",
            "eval_tag": "NO_EVAL",
        },
        {
            "role": "user",
            "content": f"<stdout>\n{create_numbered_file_content(file_lines)}\n</stdout>",
            "eval_tag": None,
        },
    ]

    # Expected response
    expected_response = create_sed_command(
        file_path=file_path,
        start_line=gap_line,
        end_line=gap_line,
        new_content=parsed["canonical_solution"],
        viewport_start=viewport_start,
        viewport_end=viewport_end,
    )

    return {
        "task_id": f"humaneval_error/{parsed['task_id']}",
        "context": context,
        "expected_final_response": expected_response,
        "humaneval_meta": {
            "original_task_id": parsed["task_id"],
            "entry_point": entry_point,
            "test": parsed["test"],
            "canonical_solution": parsed["canonical_solution"],
        },
    }


def create_continuation_testcase(
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

    # Calculate viewport
    viewport_start = max(1, partial_line_num - 10)
    viewport_end = min(len(file_lines) + len(solution_lines), partial_line_num + 10)

    # Create original file lines (before partial edit) for initial cat
    original_file_lines = parsed["prompt_lines"].copy()
    original_file_lines.append("")  # Empty gap
    original_file_lines.extend(parsed["suffix_lines"])

    # Context: ls, cat, then model already made a partial edit, now views file
    context = [
        # ls to explore
        {
            "role": "assistant",
            "content": f"```bash\nls {base_path}/\n```",
            "eval_tag": "NO_EVAL",
        },
        {
            "role": "user",
            "content": f"<stdout>\n{entry_point}.py\n</stdout>",
            "eval_tag": None,
        },
        # cat to view file
        {
            "role": "assistant",
            "content": f"```bash\ncat -n {file_path}\n```",
            "eval_tag": "NO_EVAL",
        },
        {
            "role": "user",
            "content": f"<stdout>\n{create_numbered_file_content(original_file_lines)}\n</stdout>",
            "eval_tag": None,
        },
        # Previous partial edit (simulated)
        {
            "role": "assistant",
            "content": f"```bash\nsed -i '{partial_line_num}c\\{escape_for_sed(partial_solution)}' {file_path} && cat -n {file_path} | sed -n '{viewport_start},{viewport_end}p'\n```",
            "eval_tag": "NO_EVAL",
        },
        {
            "role": "user",
            "content": f"<stdout>\n{create_numbered_file_content(file_lines[viewport_start-1:viewport_end], viewport_start)}\n</stdout>",
            "eval_tag": None,
        },
    ]

    # Expected: complete the line (replace partial with full solution)
    expected_response = create_sed_command(
        file_path=file_path,
        start_line=partial_line_num,
        end_line=partial_line_num,
        new_content=parsed["canonical_solution"],
        viewport_start=viewport_start,
        viewport_end=viewport_end,
    )

    return {
        "task_id": f"humaneval_continuation/{parsed['task_id']}",
        "context": context,
        "expected_final_response": expected_response,
        "humaneval_meta": {
            "original_task_id": parsed["task_id"],
            "entry_point": entry_point,
            "test": parsed["test"],
            "canonical_solution": parsed["canonical_solution"],
            "partial_solution": partial_solution,
        },
    }


def convert_humaneval_to_testcases(args: Args) -> List[Dict[str, Any]]:
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

    for task in tasks:
        parsed = parse_humaneval_task(task)

        # Skip tasks with empty canonical solution
        if not parsed["canonical_solution"].strip():
            continue

        if "direct" in versions_to_create:
            tc = create_direct_completion_testcase(parsed, args.base_path)
            test_cases.append(tc)

        if "error" in versions_to_create:
            tc = create_error_recovery_testcase(parsed, args.base_path)
            test_cases.append(tc)

        if "continuation" in versions_to_create:
            tc = create_continuation_testcase(parsed, args.base_path, rng)
            if tc:
                test_cases.append(tc)

    return test_cases


def main():
    args = tyro.cli(Args)

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Convert
    test_cases = convert_humaneval_to_testcases(args)

    # Write output
    with open(args.output_file, "w") as f:
        for tc in test_cases:
            f.write(json.dumps(tc) + "\n")

    print(f"\nCreated {len(test_cases)} test cases")
    print(f"Output written to: {args.output_file}")

    # Print breakdown by version
    version_counts = {}
    for tc in test_cases:
        version = tc["task_id"].split("/")[0]
        version_counts[version] = version_counts.get(version, 0) + 1

    print("\nBreakdown by version:")
    for version, count in sorted(version_counts.items()):
        print(f"  {version}: {count}")


if __name__ == "__main__":
    main()
