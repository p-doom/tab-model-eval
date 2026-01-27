import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class InputFormat(Enum):
    SED = "sed"
    ZETA = "zeta"
    UNKNOWN = "unknown"


@dataclass
class UnifiedTestCase:
    task_id: str
    format: InputFormat
    context: List[Dict[str, str]]
    expected_response: str
    assertions: Optional[str]
    input_files: Dict[str, str]
    cursor: Optional[Dict[str, Any]]
    raw: Dict[str, Any]


def detect_format(test_case: Dict[str, Any]) -> InputFormat:
    expected = test_case.get("expected_final_response", "")
    context = test_case.get("context", [])

    if "<output>" in expected and "<|editable_region" in expected:
        return InputFormat.ZETA

    if "```bash" in expected or "sed -i" in expected:
        return InputFormat.SED

    if context:
        first_content = context[0].get("content", "") if context else ""
        if "<events>" in first_content or "<input>" in first_content:
            return InputFormat.ZETA
        if "```bash" in first_content or "<stdout>" in first_content:
            return InputFormat.SED

    return InputFormat.UNKNOWN


def detect_format_from_file(jsonl_path: str) -> InputFormat:
    with open(jsonl_path, "r") as f:
        first_line = f.readline()
        if first_line:
            return detect_format(json.loads(first_line))
    return InputFormat.UNKNOWN


def _extract_files_from_sed_context(context: List[Dict[str, str]]) -> Dict[str, str]:
    files = {}
    current_file = None

    for msg in context:
        content = msg.get("content", "")
        role = msg.get("role", "")

        if role == "assistant":
            path_match = re.search(r"(?:sed -i|cat -n|cat)\s+.*?([^\s'\"]+\.\w+)", content)
            if path_match:
                current_file = path_match.group(1)

        elif role == "user" and current_file:
            stdout_match = re.search(r"<stdout>\s*(.*?)\s*</stdout>", content, re.DOTALL)
            if stdout_match:
                lines = []
                for line in stdout_match.group(1).split("\n"):
                    line_match = re.match(r"\s*\d+\t(.*)", line)
                    if line_match:
                        lines.append(line_match.group(1))
                if lines:
                    files[current_file] = "\n".join(lines)

    return files


def _extract_files_from_zeta_context(context: List[Dict[str, str]]) -> Dict[str, str]:
    files = {}

    for msg in context:
        content = msg.get("content", "")
        input_match = re.search(r"<input>\s*(.*?)\s*</input>", content, re.DOTALL)
        if input_match:
            code_match = re.search(r"```([^\n]+)\n(.*?)```", input_match.group(1), re.DOTALL)
            if code_match:
                file_path = code_match.group(1).strip()
                file_content = code_match.group(2)
                file_content = re.sub(r"<\|editable_region_start\|>\n?", "", file_content)
                file_content = re.sub(r"<\|editable_region_end\|>\n?", "", file_content)
                file_content = re.sub(r"<\|user_cursor_is_here\|>", "", file_content)
                files[file_path] = file_content

    return files


def _extract_cursor_from_zeta(context: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    for msg in context:
        content = msg.get("content", "")
        input_match = re.search(r"<input>\s*(.*?)\s*</input>", content, re.DOTALL)
        if input_match:
            code_match = re.search(r"```([^\n]+)\n(.*?)```", input_match.group(1), re.DOTALL)
            if code_match:
                file_path = code_match.group(1).strip()
                lines = code_match.group(2).split("\n")
                for line_num, line in enumerate(lines, 1):
                    cursor_pos = line.find("<|user_cursor_is_here|>")
                    if cursor_pos != -1:
                        return {"file": file_path, "line": line_num, "column": cursor_pos}
    return None


def normalize_test_case(test_case: Dict[str, Any]) -> UnifiedTestCase:
    fmt = detect_format(test_case)
    task_id = test_case.get("task_id", "unknown")
    context = test_case.get("context", [])
    expected = test_case.get("expected_final_response", "")
    assertions = test_case.get("assertions")

    if fmt == InputFormat.SED:
        input_files = _extract_files_from_sed_context(context)
        cursor = None
    elif fmt == InputFormat.ZETA:
        input_files = _extract_files_from_zeta_context(context)
        cursor = _extract_cursor_from_zeta(context)
    else:
        input_files = {}
        cursor = None

    return UnifiedTestCase(
        task_id=task_id,
        format=fmt,
        context=context,
        expected_response=expected,
        assertions=assertions,
        input_files=input_files,
        cursor=cursor,
        raw=test_case,
    )


def load_test_cases(jsonl_path: str, limit: int = -1) -> List[UnifiedTestCase]:
    test_cases = []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            test_cases.append(normalize_test_case(json.loads(line)))
    return test_cases


def get_format_stats(test_cases: List[UnifiedTestCase]) -> Dict[str, int]:
    stats = {"total": len(test_cases), "sed": 0, "zeta": 0, "unknown": 0}
    for tc in test_cases:
        if tc.format == InputFormat.SED:
            stats["sed"] += 1
        elif tc.format == InputFormat.ZETA:
            stats["zeta"] += 1
        else:
            stats["unknown"] += 1
    return stats
