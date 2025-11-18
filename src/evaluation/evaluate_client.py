import openai
import json
import os
import tyro
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class Args:
    name: str = "hello_world_insert"
    input_path: str = "data/incremental_test_cases/"
    output_path: str = "data/evaluation_results/"
    system_prompt_file: str = "data/system_prompts/minimal_v1.md"
    api_base_url: str = "http://hai005:30000/v1"


def load_dataset(filepath):
    """Yields each test case from the jsonl file."""
    with open(filepath, "r") as f:
        for line in f:
            yield json.loads(line)


def format_prompt_for_model(context, system_prompt):
    """Formats the context into a single string for the LLM."""
    prompt_str = f"SYSTEM: {system_prompt}\n"
    for message in context:
        role = message["role"].upper()
        content = message["content"]
        prompt_str += f"{role}: {content}\n"
    prompt_str += "ASSISTANT:"
    return prompt_str


def run_evaluation_client(args: Args):
    client = openai.OpenAI(base_url=args.api_base_url, api_key="EMPTY")

    test_cases = list(load_dataset(args.input_path + args.name + ".jsonl"))
    system_prompt = open(args.system_prompt_file, "r").read()

    num_correct = 0
    results = []

    print(f"Running inference on {len(test_cases)} test cases...")
    for test_case in tqdm(test_cases):
        formatted_messages = [{"role": "system", "content": system_prompt}]

        for turn in test_case["context"]:
            formatted_messages.append(turn)

        try:
            response = client.chat.completions.create(
                model="default",
                messages=formatted_messages,
                max_tokens=256,
                temperature=0.0,
                stop=["\n```", "ASSISTANT:", "USER:"],
            )
            generated_text = response.choices[0].message.content
        except Exception as e:
            print(f"API Error on task {test_case['task_id']}: {e}")
            generated_text = ""  # Mark as incorrect on error

        generated_command = generated_text
        expected_command = test_case["expected_final_response"]
        is_correct = 1 if generated_command == expected_command else 0
        num_correct += is_correct
        results.append(
            {
                "task_id": test_case["task_id"],
                "generated": generated_command,
                "expected": expected_command,
                "score": is_correct,
            }
        )

    os.makedirs(args.output_path, exist_ok=True)
    json.dump(results, open(args.output_path + args.name + ".json", "w"))
    # 6. Report final score
    pass_rate = (num_correct / len(test_cases)) * 100
    print("\n--- Evaluation Complete ---")
    print(f"Total Test Cases: {len(test_cases)}")
    print(f"Correct: {num_correct}")
    print(f"Pass Rate (Exact Match): {pass_rate:.2f}%")


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_evaluation_client(args)
