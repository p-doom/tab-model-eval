import json
import re
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    name: str = "hello_world_insert"
    input_path: str = "data/handcrafted_traces/"
    output_path: str = "data/incremental_test_cases/"


def parse_md_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    sections = re.split(r"# (User|Assistant)", content)
    sections = [s.strip() for s in sections if s.strip()]

    parsed_data = []
    current_dialogue = None

    for index, section in enumerate(sections):
        if section.startswith("User") or section.startswith("Assistant"):
            if current_dialogue is not None:
                parsed_data.append(current_dialogue)
            current_dialogue = {
                "role": section.lower(),
                "content": sections[index + 1].strip().replace("\n```", "```"),
            }

    if current_dialogue is not None:
        parsed_data.append(current_dialogue)

    return parsed_data


def create_incremental_test_cases(parsed_data, task_id):
    test_cases = []

    task_number = 0
    for i in range(1, len(parsed_data) - 1, 2):  # Ensure context ends with user content
        context = parsed_data[: i + 1]
        expected_response = (
            parsed_data[i + 1]["content"]
            if parsed_data[i + 1]["role"] == "assistant"
            else ""
        )

        test_case = {
            "task_id": f"{task_id}/{task_number}",
            "context": context,
            "expected_final_response": expected_response,
        }
        test_cases.append(test_case)
        task_number += 1

    return test_cases


def write_jsonl(test_cases, output_file):
    with open(output_file, "w") as file:
        for test_case in test_cases:
            file.write(json.dumps(test_case) + "\n")


# Main execution
if __name__ == "__main__":
    args = tyro.cli(Args)

    task_id = args.name
    input_file_path = args.input_path + args.name + ".md"
    output_file_path = args.output_path + args.name + ".jsonl"

    parsed_data = parse_md_file(input_file_path)
    incremental_test_cases = create_incremental_test_cases(parsed_data, task_id)
    write_jsonl(incremental_test_cases, output_file_path)
