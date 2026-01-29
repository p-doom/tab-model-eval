import copy
from typing import Any, Dict, List, Optional, Tuple

from crowd_pilot_serializer import convert_yaml_to_conversations

from ..types import InputFormat, get_eval_state_type
from .base import FormatConverter


def _is_edit_command(content: str) -> bool:
    return "sed -i" in content


def _is_terminal_command(content: str) -> bool:
    if "```bash" not in content and "```" not in content:
        return False
    return "sed " not in content and "cat " not in content


def _is_new_file_command(content: str) -> bool:
    if "cat -n " not in content:
        return False
    return "| sed" not in content and "sed -i" not in content


def _is_viewport_command(content: str) -> bool:
    if "sed -i" in content:
        return False
    return "cat -n " in content and "| sed -n" in content


def _find_last_message_by_type(
    messages: List[Dict[str, str]],
    eval_type: str,
) -> Tuple[int, str]:
    assistant_msgs = [(i, m) for i, m in enumerate(messages) if m["role"] == "assistant"]

    if not assistant_msgs:
        return -1, ""

    if eval_type == "file_edit":
        matching = [(i, m) for i, m in assistant_msgs if _is_edit_command(m["content"])]
    elif eval_type == "cursor_change":
        matching = [(i, m) for i, m in assistant_msgs if _is_viewport_command(m["content"])]
    elif eval_type == "terminal":
        matching = [(i, m) for i, m in assistant_msgs if _is_terminal_command(m["content"])]
    elif eval_type == "new_file":
        matching = [(i, m) for i, m in assistant_msgs if _is_new_file_command(m["content"])]
    else:
        matching = assistant_msgs

    if matching:
        idx, msg = matching[-1]
        return idx, msg["content"]
    return -1, ""


class SEDConverter(FormatConverter):
    @property
    def format_type(self) -> InputFormat:
        return InputFormat.SED

    def prepare_yaml(self, raw_yaml: Dict[str, Any], eval_idx: int) -> Dict[str, Any]:
        modified = copy.deepcopy(raw_yaml)
        modified["states"] = modified["states"][: eval_idx + 1]
        return modified

    def convert_to_messages(self, yaml_content: str) -> Optional[List[Dict[str, Any]]]:
        return convert_yaml_to_conversations(yaml_content)

    def extract_context_and_response(
        self,
        conversations: List[Dict[str, Any]],
        raw_yaml: Dict[str, Any],
        eval_idx: int,
        modified_yaml: Dict[str, Any],
    ) -> Tuple[List[Dict[str, str]], str, Dict[str, Any]]:
        messages = conversations[0]["messages"]

        if len(messages) < 2:
            return [], "", {}

        eval_type = get_eval_state_type(raw_yaml, eval_idx)
        msg_idx, expected_response = _find_last_message_by_type(messages, eval_type)

        if msg_idx < 0:
            return [], "", {}

        return messages[:msg_idx], expected_response, {}
