import json
import tyro
import os
from dataclasses import dataclass


@dataclass
class Args:
    input_file: str = "data/eval/val/validation.jsonl"
    output_file: str = "data/eval/val/validation_testcases.jsonl"
    task_name: str = "validation_set"


def convert_to_incremental_jsonl(line, output_file, task_name):
    data = json.loads(line)

    system_prompt = data.get("system_prompt", "")
    conversations = data.get("conversations", [])

    output_lines = []
    context = []

    for i, conv in enumerate(conversations):
        # Convert "Assistant"/"User" to "assistant"/"user"
        role = conv["from"].lower()
        content = conv["value"]

        # Add current message to context
        context.append({"role": role, "content": content})

        if i > 0 and role == "assistant":
            task_entry = {
                "system_prompt": system_prompt,
                "task_id": f"{task_name}/{i-1}",
                "context": context[:-1].copy(),
                "expected_final_response": content,
            }
            output_lines.append(task_entry)

    with open(output_file, "a") as f:
        for line in output_lines:
            f.write(json.dumps(line) + "\n")

    print(f"Converted {len(output_lines)} entries to {output_file}")


if __name__ == "__main__":
    args = tyro.cli(Args)

    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    # iterate over jsonl file and convert to incremental jsonl
    conversation_id = 0
    with open(args.input_file, "r") as f:
        for line in f:
            task_name = f"conversation_{conversation_id}/{args.task_name}"
            convert_to_incremental_jsonl(line, args.output_file, task_name)
            conversation_id += 1
