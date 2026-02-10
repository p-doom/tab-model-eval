"""Shared types and utilities for evaluation."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class InputFormat(Enum):
    SED = "sed"
    ZETA = "zeta"


@dataclass
class TestCaseWithYaml:
    task_id: str
    context: List[Dict[str, str]]
    expected_response: str
    assertions: Optional[str]
    input_files: Dict[str, str]
    raw_yaml: Dict[str, Any]
    format: InputFormat
    editable_range: Optional[Dict[str, int]] = None
    editable_file: Optional[str] = None
    unsupported: bool = False


def find_all_eval_state_indices(raw_yaml: Dict[str, Any]) -> List[int]:
    states = raw_yaml.get("states", [])
    indices = []
    for i, state in enumerate(states):
        if state.get("eval", "NO_EVAL") == "EVAL":
            indices.append(i)
    return indices


def get_eval_state_type(raw_yaml: Dict[str, Any], state_idx: int) -> str:
    states = raw_yaml.get("states", [])
    if state_idx >= len(states):
        return "unknown"

    state = states[state_idx]

    terminal = state.get("terminal")
    if terminal and terminal.get("command"):
        return "terminal"

    if state_idx > 0:
        prev_files = states[state_idx - 1].get("files") or {}
        curr_files = state.get("files") or {}

        for file_path, content in curr_files.items():
            if file_path in prev_files:
                if prev_files[file_path] != content:
                    return "file_edit"
            else:
                return "new_file"

        prev_cursor = states[state_idx - 1].get("cursor")
        curr_cursor = state.get("cursor")
        if curr_cursor and curr_cursor.get("file"):
            if not prev_cursor or prev_cursor.get("file") != curr_cursor.get("file"):
                return "cursor_change"
            elif prev_cursor.get("line") != curr_cursor.get("line"):
                return "cursor_change"
    elif state_idx == 0:
        curr_files = state.get("files") or {}
        if curr_files:
            return "new_file"

    return "unknown"


def get_input_files_for_eval_step(raw_yaml: Dict[str, Any], eval_idx: int) -> Dict[str, str]:
    states = raw_yaml.get("states", [])
    files: Dict[str, str] = {}

    for i in range(eval_idx):
        if i >= len(states):
            break
        state_files = states[i].get("files")
        if state_files:
            files.update(state_files)

    return files


def get_assertions_for_eval_step(raw_yaml: Dict[str, Any], eval_idx: int) -> Optional[str]:
    states = raw_yaml.get("states", [])
    if eval_idx < len(states):
        return states[eval_idx].get("judge_assertions")
    return None
