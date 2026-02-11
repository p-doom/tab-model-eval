import copy
from typing import Any, Dict, List, Optional, Tuple

from crowd_pilot_serializer import convert_yaml_to_zeta, zeta_system_prompt

from src.utils.types import InputFormat, get_input_files_for_eval_step
from src.formats.base import FormatConverter


def _get_cursor_file_from_yaml(raw_yaml: Dict[str, Any]) -> Optional[str]:
    states = raw_yaml.get("states", [])
    for state in states:
        if state.get("eval", "NO_EVAL") == "EVAL":
            cursor = state.get("cursor") or {}
            return cursor.get("file")
    return None


class ZetaConverter(FormatConverter):
    @property
    def format_type(self) -> InputFormat:
        return InputFormat.ZETA

    def prepare_yaml(self, raw_yaml: Dict[str, Any], eval_idx: int) -> Dict[str, Any]:
        modified = copy.deepcopy(raw_yaml)
        for i, state in enumerate(modified.get("states", [])):
            if state.get("eval") == "EVAL" and i != eval_idx:
                state["eval"] = "NO_EVAL"
        return modified

    def convert_to_messages(self, yaml_content: str) -> Optional[List[Dict[str, Any]]]:
        return convert_yaml_to_zeta(yaml_content)

    def extract_context_and_response(
        self,
        conversations: List[Dict[str, Any]],
        raw_yaml: Dict[str, Any],
        eval_idx: int,
        modified_yaml: Dict[str, Any],
    ) -> Tuple[List[Dict[str, str]], str, Dict[str, Any]]:
        conv = conversations[0]
        messages = conv["messages"]
        editable_range = conv.get("editable_range")

        if len(messages) < 2:
            return [], "", {}

        system_msg = messages[0]
        expected_msg = messages[1]

        zeta_prefix = zeta_system_prompt()
        system_content = system_msg.get("content", "")
        if system_content.startswith(zeta_prefix):
            context_text = system_content[len(zeta_prefix) :].lstrip()
        else:
            context_text = system_content

        context = [{"role": "user", "content": context_text}]
        expected_response = expected_msg["content"]

        input_files = get_input_files_for_eval_step(raw_yaml, eval_idx)
        editable_file = _get_cursor_file_from_yaml(modified_yaml)
        if editable_file is None and len(input_files) == 1:
            editable_file = next(iter(input_files.keys()))

        extra_fields = {
            "editable_range": editable_range,
            "editable_file": editable_file,
        }

        return context, expected_response, extra_fields
