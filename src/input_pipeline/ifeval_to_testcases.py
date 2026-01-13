#!/usr/bin/env python3
"""
Convert IFEval benchmark data to eval test cases format.

IFEval (Instruction Following Eval) tests whether models can follow
verifiable instructions like:
- "write at least 400 words"
- "do not use any commas"
- "mention 'AI' at least 3 times"
- "respond in all lowercase"

Usage:
    python ifeval_to_testcases.py \
        --input_file /path/to/ifeval_input_data.jsonl \
        --output_file /path/to/ifeval_testcases.jsonl

    # Generate only a subset for testing
    python src/input_pipeline/ifeval_to_testcases.py \
        --input_file /fast/project/HFMI_SynergyUnit/tab_model/huggingface/IFEval/ifeval_input_data.jsonl \
        --output_file data/eval/ifeval/ifeval_testcases.jsonl \
        --limit 50
"""

import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any

import tyro


@dataclass
class Args:
    input_file: str = (
        "/fast/project/HFMI_SynergyUnit/tab_model/huggingface/IFEval/ifeval_input_data.jsonl"
    )
    output_file: str = "data/eval/ifeval/ifeval_testcases.jsonl"
    # Limit number of tasks (for testing), -1 for all
    limit: int = -1
    # Whether to include a system prompt context
    include_system_context: bool = True
    # Optional system prompt to prepend
    system_prompt: str = "You are a helpful assistant that follows instructions precisely."


def create_ifeval_testcase(
    sample: Dict[str, Any],
    include_system_context: bool = True,
    system_prompt: str = "",
) -> Dict[str, Any]:
    """
    Convert an IFEval sample to our testcase format.

    IFEval sample format:
    {
        "key": 1000,
        "prompt": "Write a 300+ word summary...",
        "instruction_id_list": ["punctuation:no_comma", "length_constraints:number_words"],
        "kwargs": [{}, {"num_words": 300, "relation": "at least"}]
    }

    Output testcase format:
    {
        "task_id": "ifeval_1000",
        "context": [...],
        "expected_final_response": "",  # IFEval doesn't have a fixed expected response
        "ifeval_meta": {
            "key": 1000,
            "prompt": "...",
            "instruction_id_list": [...],
            "kwargs": [...]
        }
    }
    """
    key = sample.get("key", 0)
    prompt = sample.get("prompt", "")
    instruction_id_list = sample.get("instruction_id_list", [])
    kwargs = sample.get("kwargs", [])

    # Create context - the prompt is presented as a user message
    context = []

    if include_system_context and system_prompt:
        context.append(
            {
                "role": "system",
                "content": system_prompt,
                "eval_tag": None,
            }
        )

    # The actual instruction prompt from IFEval
    context.append(
        {
            "role": "user",
            "content": prompt,
            "eval_tag": None,
        }
    )

    return {
        "task_id": f"ifeval_{key}",
        "context": context,
        # IFEval doesn't have a single correct answer - evaluation is based on
        # whether the response follows all the specified instructions
        "expected_final_response": "",
        "ifeval_meta": {
            "key": key,
            "prompt": prompt,
            "instruction_id_list": instruction_id_list,
            "kwargs": kwargs,
        },
    }


def convert_ifeval_to_testcases(args: Args) -> List[Dict[str, Any]]:
    """Main conversion function."""

    # Load IFEval samples
    samples = []
    with open(args.input_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if args.limit > 0:
        samples = samples[: args.limit]

    print(f"Loaded {len(samples)} IFEval samples")

    # Convert each sample
    test_cases = []
    instruction_type_counts: Dict[str, int] = {}

    for sample in samples:
        tc = create_ifeval_testcase(
            sample,
            include_system_context=args.include_system_context,
            system_prompt=args.system_prompt,
        )
        test_cases.append(tc)

        # Count instruction types for statistics
        for instr_id in sample.get("instruction_id_list", []):
            category = instr_id.split(":")[0] if ":" in instr_id else instr_id
            instruction_type_counts[category] = instruction_type_counts.get(category, 0) + 1

    # Print instruction type statistics
    print("\nInstruction type breakdown:")
    for category, count in sorted(instruction_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {category}: {count}")

    return test_cases


def main():
    args = tyro.cli(Args)

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Convert
    test_cases = convert_ifeval_to_testcases(args)

    # Write output
    with open(args.output_file, "w") as f:
        for tc in test_cases:
            f.write(json.dumps(tc) + "\n")

    print(f"\nCreated {len(test_cases)} test cases")
    print(f"Output written to: {args.output_file}")

    # Print sample testcase
    if test_cases:
        print("\n--- Sample testcase ---")
        print(json.dumps(test_cases[0], indent=2))


if __name__ == "__main__":
    main()
