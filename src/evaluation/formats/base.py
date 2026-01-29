from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..types import (
    InputFormat,
    TestCaseWithYaml,
    find_all_eval_state_indices,
    get_assertions_for_eval_step,
    get_input_files_for_eval_step,
    get_eval_state_type,
)


class FormatConverter(ABC):
    @property
    @abstractmethod
    def format_type(self) -> InputFormat:
        pass

    @abstractmethod
    def prepare_yaml(self, raw_yaml: Dict[str, Any], eval_idx: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def convert_to_messages(self, yaml_content: str) -> Optional[List[Dict[str, Any]]]:
        pass

    @abstractmethod
    def extract_context_and_response(
        self,
        conversations: List[Dict[str, Any]],
        raw_yaml: Dict[str, Any],
        eval_idx: int,
        modified_yaml: Dict[str, Any],
    ) -> Tuple[List[Dict[str, str]], str, Dict[str, Any]]:
        pass

    def create_unsupported_test_case(
        self,
        task_id: str,
        raw_yaml: Dict[str, Any],
        eval_idx: int,
    ) -> TestCaseWithYaml:
        return TestCaseWithYaml(
            task_id=task_id,
            context=[],
            expected_response="",
            assertions=get_assertions_for_eval_step(raw_yaml, eval_idx),
            input_files={},
            raw_yaml=raw_yaml,
            format=self.format_type,
            unsupported=True,
        )

    def convert(self, yaml_data_list: List[Dict[str, Any]]) -> List[TestCaseWithYaml]:
        results = []

        for raw_yaml in yaml_data_list:
            base_task_id = raw_yaml.get("task_id", "unknown")
            eval_indices = find_all_eval_state_indices(raw_yaml)

            if not eval_indices:
                print(f"Warning: No EVAL step found for {base_task_id}, skipping")
                continue

            for eval_idx in eval_indices:
                task_id = (
                    f"{base_task_id}/step_{eval_idx}" if len(eval_indices) > 1 else base_task_id
                )

                modified_yaml = self.prepare_yaml(raw_yaml, eval_idx)
                yaml_content = yaml.dump(modified_yaml)

                conversations = self.convert_to_messages(yaml_content)

                if not conversations:
                    print(f"Warning: No conversations generated for {task_id}")
                    results.append(self.create_unsupported_test_case(task_id, raw_yaml, eval_idx))
                    continue

                context, expected_response, extra_fields = self.extract_context_and_response(
                    conversations, raw_yaml, eval_idx, modified_yaml
                )

                if not expected_response:
                    eval_type = get_eval_state_type(raw_yaml, eval_idx)
                    print(f"Warning: Could not find {eval_type} message for {task_id}, skipping")
                    continue

                input_files = get_input_files_for_eval_step(raw_yaml, eval_idx)
                assertions = get_assertions_for_eval_step(raw_yaml, eval_idx)

                results.append(
                    TestCaseWithYaml(
                        task_id=task_id,
                        context=context,
                        expected_response=expected_response,
                        assertions=assertions,
                        input_files=input_files,
                        raw_yaml=raw_yaml,
                        format=self.format_type,
                        **extra_fields,
                    )
                )

        return results
