import json
import os
import re
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    input_dir: str = "data/eval/handcrafted"
    output_file: str = "data/eval/handcrafted_test_cases/handcrafted_test_cases.jsonl"


def parse_md_file(input_file):
    with open(input_file, "r") as file:
        content = file.read()

    # Use regex to find all headers with optional eval tags
    # Pattern matches: # User or # Assistant <EVAL> or # Assistant <NO_EVAL>
    pattern = r"# (User|Assistant)(?: <(EVAL|NO_EVAL)>)?\n"

    parsed_data = []

    # Find all matches
    for match in re.finditer(pattern, content):
        # Get the role and eval_tag from the match
        role = match.group(1).lower()
        eval_tag = match.group(2)  # Will be None if not present

        # Find the content between this match and the next match (or end of file)
        start_pos = match.end()
        next_match = None
        for next_match_iter in re.finditer(pattern, content):
            if next_match_iter.start() > match.start():
                next_match = next_match_iter
                break

        if next_match:
            end_pos = next_match.start()
        else:
            end_pos = len(content)

        dialogue_content = content[start_pos:end_pos].strip()

        parsed_data.append(
            {
                "role": role,
                "content": dialogue_content,
                "eval_tag": eval_tag,  # None for User, "EVAL" or "NO_EVAL" for Assistant
            }
        )

    return parsed_data


def create_incremental_test_cases(parsed_data, task_name):
    test_cases = []

    task_number = 0
    for i in range(1, len(parsed_data) - 1, 2):  # Ensure context ends with user content
        # Only create test cases for assistant turns marked with <EVAL>
        if i + 1 < len(parsed_data):
            assistant_turn = parsed_data[i + 1]
            if assistant_turn["role"] == "assistant":
                eval_tag = assistant_turn.get("eval_tag")
                if eval_tag != "EVAL":
                    continue

                context = parsed_data[: i + 1]
                expected_response = assistant_turn["content"]

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

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    incremental_test_cases_total = []
    # iterate over all md files in the data/eval/handcrafted directory
    for file in os.listdir(args.input_dir):
        if file.endswith(".md"):
            task_name = file.split("/")[-1].split(".")[0]
            parsed_data = parse_md_file(os.path.join(args.input_dir, file))
            incremental_test_cases = create_incremental_test_cases(parsed_data, task_name)
            incremental_test_cases_total.extend(incremental_test_cases)

    write_jsonl(incremental_test_cases_total, args.output_file)
    print(
        f"Created {len(incremental_test_cases_total)} incremental test cases and saved to {args.output_file}"
    )
