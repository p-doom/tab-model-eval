import json
import re
import tyro
from dataclasses import dataclass

@dataclass
class Args:
    input_file_path: str = "data/handcrafted_traces/insert_hello_world.md"
    output_file_path: str = "data/incremental_test_cases/"
    name: str = "insert_hello_world"


def parse_md_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        
    sections = re.split(r'# (User|Assistant)', content)
    sections = [s.strip() for s in sections if s.strip()]

    parsed_data = []
    current_dialogue = None

    for index, section in enumerate(sections):
        if section.startswith('User') or section.startswith('Assistant'):
            if current_dialogue is not None:
                parsed_data.append(current_dialogue)
            current_dialogue = {'role': section.lower(), 'content': sections[index + 1].strip()}
            
    if current_dialogue is not None:
        parsed_data.append(current_dialogue)

    return parsed_data

def create_incremental_test_cases(parsed_data):
    test_cases = []

    for i in range(1, len(parsed_data) - 1, 2):  # Ensure context ends with user content
        context = parsed_data[:i + 1]
        expected_response = parsed_data[i + 1]['content'] if parsed_data[i + 1]['role'] == 'assistant' else ""
        
        test_case = {
            "context": context,
            "expected_final_response": expected_response,
            "metric": "correct_response"
        }
        test_cases.append(test_case)

    return test_cases

def write_jsonl(test_cases, output_file):
    with open(output_file, 'w') as file:
        for test_case in test_cases:
            file.write(json.dumps(test_case) + '\n')

if __name__ == "__main__":
    args = Args()
    

    parsed_data = parse_md_file(args.input_file_path)
    incremental_test_cases = create_incremental_test_cases(parsed_data)
    write_jsonl(incremental_test_cases, args.output_file_path + args.name + ".jsonl")
    print(f'Incremental test cases saved to {args.output_file_path + args.name + ".jsonl"}')   
