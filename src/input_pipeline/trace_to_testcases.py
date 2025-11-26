import json
import re
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    input_file: str = "data/eval/hello_world_insert.md"
    output_file: str = "data/eval/hello_world_insert.jsonl"


def parse_md_file(input_file):
    with open(input_file, "r") as file:
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
                "content": sections[index + 1].strip(),
            }

    if current_dialogue is not None:
        parsed_data.append(current_dialogue)

    return parsed_data


def create_incremental_test_cases(parsed_data, task_name):
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
            "task_id": f"{task_name}/{task_number}",
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

    task_name = args.input_file.split("/")[-1].split(".")[0]
    parsed_data = parse_md_file(args.input_file)
    incremental_test_cases = create_incremental_test_cases(parsed_data, task_name)
    write_jsonl(incremental_test_cases, args.output_file)
    print(
        f"Created {len(incremental_test_cases)} incremental test cases and saved to {args.output_file}"
    )
