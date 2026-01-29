import asyncio
import glob
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
import tyro
import yaml
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from crowd_pilot_serializer import (
    convert_yaml_to_conversations,
    convert_yaml_to_zeta,
    default_system_prompt,
    zeta_system_prompt,
)

from .prediction_applicators import (
    apply_sed_prediction,
    apply_zeta_prediction_to_editable,
    parse_viewport_command,
    parse_zeta_cursor_position,
)


class InputFormat(Enum):
    SED = "sed"
    ZETA = "zeta"


@dataclass
class Args:
    input_format: str = ""  # "sed" or "zeta"
    yaml_input_dir: str = "data/eval/handcrafted"
    output_file: str = "data/eval/generations/generations.yaml"
    limit: int = -1

    system_prompt_file: str = ""
    viewport_radius: int = 10

    model_name: str = "default"
    model_path: str = ""
    presence_penalty: float = 1.5
    num_samples: int = 5
    max_new_tokens: int = 2048

    server_host: str = "0.0.0.0"
    server_port: int = 30000
    context_length: int = 8192
    problem_length: int = 8192
    mem_fraction_static: float = 0.95
    api_key: str = "EMPTY"
    tp_size: int = 1
    dp_size: int = 1
    lora_paths: Optional[List[str]] = None
    launch_server: bool = True
    extra_server_args: Optional[List[str]] = None
    extra_server_args_str: str = ""

    concurrency: int = 16
    max_connections: int = 256
    keepalive: int = 60
    max_attempts: int = 6
    timeout: float = 300.0


@dataclass
class TestCaseWithYaml:
    """Holds both the converted test case and raw YAML data."""

    task_id: str
    context: List[Dict[str, str]]
    expected_response: str
    assertions: Optional[str]
    input_files: Dict[str, str]  # file state before EVAL
    raw_yaml: Dict[str, Any]  # original YAML for output
    format: InputFormat
    editable_range: Optional[Dict[str, int]] = None
    editable_file: Optional[str] = None


def load_yaml_files(yaml_dir: str, limit: int = -1) -> List[Dict[str, Any]]:
    """Load all YAML files from directory."""
    pattern = os.path.join(yaml_dir, "*.yaml")
    yaml_files = sorted(glob.glob(pattern))

    if limit > 0:
        yaml_files = yaml_files[:limit]

    results = []
    for path in yaml_files:
        with open(path, "r") as f:
            results.append(yaml.safe_load(f))

    return results


def get_input_files_from_yaml(raw_yaml: Dict[str, Any]) -> Dict[str, str]:
    """Extract the file state just before the EVAL step."""
    states = raw_yaml.get("states", [])

    # Find the state just before EVAL
    files = {}
    for i, state in enumerate(states):
        eval = state.get("eval", "NO_EVAL")
        if eval == "EVAL":
            # Return files from previous state
            if i > 0:
                prev_files = states[i - 1].get("files", {})
                if prev_files:
                    files = prev_files
            break
        # Keep track of latest files
        state_files = state.get("files", {})
        if state_files:
            files.update(state_files)

    return files


def get_input_files_for_eval(
    raw_yaml: Dict[str, Any],
) -> Dict[str, str]:
    """Get input files for evaluation from YAML states only."""
    files: Dict[str, str] = get_input_files_from_yaml(raw_yaml)
    if not files:
        raise ValueError("file_state_not_found_in_yaml")
    return files


def get_assertions_from_yaml(raw_yaml: Dict[str, Any]) -> Optional[str]:
    """Extract judge_assertions from the EVAL state."""
    states = raw_yaml.get("states", [])
    for state in states:
        eval = state.get("eval", "NO_EVAL")
        if eval == "EVAL":
            return state.get("judge_assertions")
    return None


def get_cursor_file_from_yaml(raw_yaml: Dict[str, Any]) -> Optional[str]:
    states = raw_yaml.get("states", [])
    for state in states:
        eval = state.get("eval", state.get("eval", "NO_EVAL"))
        if eval == "EVAL":
            cursor = state.get("cursor") or {}
            return cursor.get("file")
    return None


def find_first_eval_state_index(raw_yaml: Dict[str, Any]) -> int:
    """Find the index of the first EVAL state."""
    states = raw_yaml.get("states", [])
    for i, state in enumerate(states):
        eval = state.get("eval", "NO_EVAL")
        if eval == "EVAL":
            return i
    return -1


def find_eval_message_split(
    messages: List[Dict[str, str]],
    raw_yaml: Dict[str, Any],
) -> tuple[List[Dict[str, str]], str]:
    """
    Find where to split messages for evaluation.

    Returns (context_messages, expected_response) where:
    - context_messages: all messages before the first EVAL edit
    - expected_response: the assistant message at the first EVAL step
    """
    first_eval_idx = find_first_eval_state_index(raw_yaml)
    if first_eval_idx <= 0:
        # No valid EVAL state found
        return [], ""

    assistant_idxs = [i for i, msg in enumerate(messages) if msg["role"] == "assistant"]
    if first_eval_idx >= len(assistant_idxs):
        return [], ""
    msg_idx = assistant_idxs[first_eval_idx]
    return messages[:msg_idx], messages[msg_idx]["content"]


def convert_yaml_to_test_cases_sed(
    yaml_data_list: List[Dict[str, Any]],
) -> List[TestCaseWithYaml]:
    """Convert YAML data to SED-format test cases."""
    results = []

    for raw_yaml in yaml_data_list:
        yaml_content = yaml.dump(raw_yaml)
        task_id = raw_yaml.get("task_id", "unknown")

        conversations = convert_yaml_to_conversations(yaml_content)

        if not conversations:
            print(f"Warning: No conversations generated for {task_id}")
            continue

        conv = conversations[0]
        messages = conv["messages"]

        if len(messages) < 2:
            print(f"Warning: Not enough messages for {task_id}")
            continue

        context, expected_response = find_eval_message_split(messages, raw_yaml)
        if not expected_response:
            print(f"Warning: No EVAL step found for {task_id}, skipping")
            continue
        input_files = get_input_files_for_eval(raw_yaml)
        assertions = get_assertions_from_yaml(raw_yaml)

        results.append(
            TestCaseWithYaml(
                task_id=task_id,
                context=context,
                expected_response=expected_response,
                assertions=assertions,
                input_files=input_files,
                raw_yaml=raw_yaml,
                format=InputFormat.SED,
            )
        )

    return results


def convert_yaml_to_test_cases_zeta(
    yaml_data_list: List[Dict[str, Any]],
) -> List[TestCaseWithYaml]:
    """Convert YAML data to Zeta-format test cases."""
    results = []

    for raw_yaml in yaml_data_list:
        yaml_content = yaml.dump(raw_yaml)
        task_id = raw_yaml.get("task_id", "unknown")
        if find_first_eval_state_index(raw_yaml) < 0:
            print(f"Warning: No EVAL step found for {task_id}, skipping")
            continue

        # Use Zeta conversion API
        conversations = convert_yaml_to_zeta(yaml_content)

        if not conversations:
            print(f"Warning: No Zeta conversations generated for {task_id}")
            continue

        # Zeta returns [system prompt, expected output]
        conv = conversations[0]
        messages = conv["messages"]
        editable_range = conv.get("editable_range")

        if len(messages) < 2:
            print(f"Warning: Not enough messages for {task_id}")
            continue

        # For Zeta: system message is full prompt; strip the system prompt prefix
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

        # Get input files from YAML (files at EVAL step minus 1)
        input_files = get_input_files_from_yaml(raw_yaml)
        assertions = get_assertions_from_yaml(raw_yaml)
        editable_file = get_cursor_file_from_yaml(raw_yaml)
        if editable_file is None and len(input_files) == 1:
            editable_file = next(iter(input_files.keys()))

        results.append(
            TestCaseWithYaml(
                task_id=task_id,
                context=context,
                expected_response=expected_response,
                assertions=assertions,
                input_files=input_files,
                raw_yaml=raw_yaml,
                format=InputFormat.ZETA,
                editable_range=editable_range,
                editable_file=editable_file,
            )
        )

    return results


def convert_yaml_to_test_cases(
    yaml_data_list: List[Dict[str, Any]],
    input_format: InputFormat,
) -> List[TestCaseWithYaml]:
    """Convert YAML data to test cases using PyO3 bindings."""
    if input_format == InputFormat.SED:
        return convert_yaml_to_test_cases_sed(yaml_data_list)
    elif input_format == InputFormat.ZETA:
        return convert_yaml_to_test_cases_zeta(yaml_data_list)
    else:
        raise ValueError(f"Unsupported format: {input_format.value}")


def estimate_token_count(messages: List[Dict[str, str]]) -> int:
    total_chars = sum(len(msg.get("content", "")) for msg in messages)
    return total_chars // 3


def filter_by_context_length(
    test_cases: List[TestCaseWithYaml],
    system_prompt: str,
    max_context_length: int,
    problem_length: int,
    buffer_tokens: int = 512,
) -> tuple[List[TestCaseWithYaml], List[TestCaseWithYaml]]:
    valid = []
    skipped = []

    for tc in test_cases:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(tc.context)

        estimated = estimate_token_count(messages) + buffer_tokens

        if estimated <= max_context_length and estimated <= problem_length:
            valid.append(tc)
        else:
            print(f"Skipping {tc.task_id}: {estimated} tokens exceeds limit")
            skipped.append(tc)

    return valid, skipped


def extract_response_content(response_text: str, format_type: InputFormat) -> str:
    if format_type == InputFormat.SED:
        match = re.search(r"(```bash\s+.*?```)", response_text, re.DOTALL)
        return match.group(1) if match else ""
    elif format_type == InputFormat.ZETA:
        # Zeta uses 5 backticks for code blocks
        match = re.search(r"`````\n(.*?)\n`````", response_text, re.DOTALL)
        if match:
            return match.group(1)
        # Fallback to 3 backticks
        match = re.search(r"```\n(.*?)\n```", response_text, re.DOTALL)
        return match.group(1) if match else response_text
    return response_text


def apply_prediction(
    format_type: InputFormat,
    input_files: Dict[str, str],
    generated: str,
    editable_range: Optional[Dict[str, int]] = None,
    editable_file: Optional[str] = None,
) -> tuple[Optional[Dict[str, str]], Optional[str]]:
    """Apply model prediction to input files."""
    if format_type == InputFormat.SED:
        return apply_sed_prediction(input_files, generated)
    elif format_type == InputFormat.ZETA:
        return apply_zeta_prediction_to_editable(
            input_files, generated, editable_range, editable_file
        )
    return None, f"Unsupported format: {format_type.value}"


async def generate_for_test_case(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    system_prompt: str,
    test_case: TestCaseWithYaml,
    args: Args,
) -> Dict[str, Any]:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(test_case.context)

    async with sem:
        delay = 0.25
        for attempt in range(args.max_attempts):
            try:
                resp = await client.chat.completions.create(
                    model=args.model_name,
                    messages=messages,
                    presence_penalty=args.presence_penalty,
                    n=args.num_samples,
                    max_tokens=args.max_new_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )

                samples = []
                for idx, choice in enumerate(resp.choices):
                    response_text = choice.message.content or ""
                    generated = extract_response_content(response_text, test_case.format)
                    exact_match = int(generated == test_case.expected_response)
                    predicted_files, error = apply_prediction(
                        test_case.format,
                        test_case.input_files,
                        generated,
                        editable_range=test_case.editable_range,
                        editable_file=test_case.editable_file,
                    )
                    predicted_cursor = None
                    if test_case.format == InputFormat.SED:
                        viewport = parse_viewport_command(generated)
                        if viewport:
                            line = (viewport["start"] + viewport["end"]) // 2
                            predicted_cursor = {
                                "file": viewport["file_path"],
                                "line": line,
                                "column": 0,
                            }
                    elif test_case.format == InputFormat.ZETA:
                        predicted_cursor = parse_zeta_cursor_position(
                            generated, test_case.editable_range, test_case.editable_file
                        )

                    samples.append(
                        {
                            "sample_idx": idx,
                            "response_text": response_text,
                            "generated": generated,
                            "exact_match": exact_match,
                            "predicted_files": predicted_files if error is None else None,
                            "prediction_error": error,
                            "predicted_cursor": predicted_cursor,
                        }
                    )

                return {
                    "task_id": test_case.task_id,
                    "format": test_case.format.value,
                    "input_files": test_case.input_files,
                    "raw_yaml": test_case.raw_yaml,
                    "expected_response": test_case.expected_response,
                    "assertions": test_case.assertions,
                    "samples": samples,
                    "error": None,
                }

            except Exception as e:
                print(f"Error on {test_case.task_id} (attempt {attempt + 1}): {e}")
                if attempt == args.max_attempts - 1:
                    return {
                        "task_id": test_case.task_id,
                        "format": test_case.format.value,
                        "input_files": test_case.input_files,
                        "raw_yaml": test_case.raw_yaml,
                        "expected_response": test_case.expected_response,
                        "assertions": test_case.assertions,
                        "samples": [],
                        "error": str(e),
                    }
                await asyncio.sleep(delay)
                delay *= 2

    return {"task_id": test_case.task_id, "error": "max_attempts_exceeded", "samples": []}


def build_yaml_result(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    task_id = raw_result["task_id"]
    format_type = raw_result["format"]
    raw_yaml = raw_result.get("raw_yaml", {})
    samples_raw = raw_result.get("samples", [])

    # Get states directly from raw YAML (full history)
    yaml_states = raw_yaml.get("states", [])

    # Find eval step index
    eval_step = 0
    for i, state in enumerate(yaml_states):
        eval = state.get("eval", "NO_EVAL")
        if eval == "EVAL":
            eval_step = i
            break

    # Build output states from YAML states
    states = []
    for i, state in enumerate(yaml_states):
        eval = state.get("eval", "NO_EVAL")
        states.append(
            {
                "step": i,
                "eval": eval,
                "files": state.get("files", {}),
                "cursor": state.get("cursor"),
                "terminal": state.get("terminal"),
                "judge_assertions": state.get("judge_assertions") if eval == "EVAL" else None,
            }
        )

    # Build sample predictions
    samples = []
    for s in samples_raw:
        samples.append(
            {
                "sample_idx": s["sample_idx"],
                "step": eval_step,
                "predicted_files": s.get("predicted_files"),
                "predicted_cursor": s.get("predicted_cursor"),
                "predicted_raw": s.get("generated", ""),
                "exact_match": s.get("exact_match", 0),
                "prediction_error": s.get("prediction_error"),
            }
        )

    # Compute metrics
    num_samples = len(samples)
    num_exact = sum(1 for s in samples if s.get("exact_match", 0) == 1)
    pass_at_1 = 1 if num_exact > 0 else 0

    return {
        "task_id": task_id,
        "format": format_type,
        "description": raw_yaml.get("description", ""),
        "states": states,
        "samples": samples,
        "metrics": {
            "num_samples": num_samples,
            "num_exact_matches": num_exact,
            "pass_at_1": pass_at_1,
        },
    }


def validate_format(args: Args) -> InputFormat:
    if args.input_format == "sed":
        return InputFormat.SED
    elif args.input_format == "zeta":
        return InputFormat.ZETA
    else:
        raise ValueError(f"--input-format must be 'sed' or 'zeta', got '{args.input_format}'")


def write_yaml_output(
    output_file: str, results: List[Dict[str, Any]], config: Dict[str, Any]
) -> None:
    """Write generation results to YAML file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    output = {
        "config": config,
        "results": results,
    }

    with open(output_file, "w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


async def run_generation(args: Args, base_url: str):
    input_format = validate_format(args)

    print(f"Loading YAML files from {args.yaml_input_dir}...")
    yaml_data_list = load_yaml_files(args.yaml_input_dir, limit=args.limit)
    print(f"Loaded {len(yaml_data_list)} YAML files")

    print(f"Converting to {input_format.value} format test cases...")
    test_cases = convert_yaml_to_test_cases(yaml_data_list, input_format)
    print(f"Generated {len(test_cases)} test cases")

    if not args.system_prompt_file:
        raise ValueError("--system-prompt-file is required")
    if not os.path.exists(args.system_prompt_file):
        raise FileNotFoundError(f"System prompt file not found: {args.system_prompt_file}")
    print(f"Using system prompt from file: {args.system_prompt_file}")
    with open(args.system_prompt_file, "r") as f:
        system_prompt = f.read()
    if "{viewport_lines}" in system_prompt:
        viewport_lines = 2 * args.viewport_radius + 1
        system_prompt = system_prompt.format(viewport_lines=viewport_lines)

    test_cases, skipped = filter_by_context_length(
        test_cases, system_prompt, args.context_length, args.problem_length
    )
    print(f"After filtering: {len(test_cases)} valid, {len(skipped)} skipped")

    http = httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(args.timeout, connect=10.0, read=args.timeout),
        limits=httpx.Limits(
            max_connections=args.max_connections,
            max_keepalive_connections=args.max_connections,
            keepalive_expiry=args.keepalive,
        ),
        headers={"Connection": "keep-alive"},
    )
    client = AsyncOpenAI(base_url=base_url, api_key=args.api_key, http_client=http)
    sem = asyncio.Semaphore(args.concurrency)

    tasks = [generate_for_test_case(client, sem, system_prompt, tc, args) for tc in test_cases]

    print(f"Running {len(tasks)} generations with concurrency={args.concurrency}...")
    raw_results: List[Dict[str, Any]] = []
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        raw_results.append(await coro)

    await http.aclose()

    raw_results.sort(key=lambda x: x["task_id"])
    yaml_results = [build_yaml_result(r) for r in raw_results]

    print(f"Writing output to {args.output_file}...")
    write_yaml_output(args.output_file, yaml_results, config=args.__dict__)

    total_samples = sum(r["metrics"]["num_samples"] for r in yaml_results)
    total_exact = sum(r["metrics"]["num_exact_matches"] for r in yaml_results)
    total_pass = sum(r["metrics"]["pass_at_1"] for r in yaml_results)

    print("\n" + "=" * 50)
    print("Generation Complete")
    print("=" * 50)
    print(f"Tasks: {len(yaml_results)}")
    print(f"Samples: {total_samples}")
    print(f"Exact matches: {total_exact}")
    print(f"Pass@1: {total_pass}/{len(yaml_results)} ({100*total_pass/len(yaml_results):.1f}%)")
    print(f"Output: {args.output_file}")


async def wait_for_server(base_url: str, timeout: float = 300.0) -> None:
    print(f"Waiting for server at {base_url}...")
    deadline = asyncio.get_event_loop().time() + timeout

    async with httpx.AsyncClient() as client:
        while True:
            if asyncio.get_event_loop().time() > deadline:
                raise RuntimeError(f"Server not ready within {timeout}s")
            try:
                resp = await client.get(f"{base_url}/v1/models", timeout=5.0)
                if resp.status_code == 200:
                    print("Server is up.")
                    return
            except Exception as e:
                print(f"Waiting... ({e})")
            await asyncio.sleep(10.0)


def launch_sglang_server(args: Args) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--host",
        args.server_host,
        "--port",
        str(args.server_port),
        "--context-length",
        str(args.context_length),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--tp-size",
        str(args.tp_size),
        "--dp-size",
        str(args.dp_size),
    ]

    if args.lora_paths:
        cmd.append("--lora-paths")
        cmd.extend(args.lora_paths)

    if args.extra_server_args_str:
        cmd.extend(shlex.split(args.extra_server_args_str))

    if args.extra_server_args:
        cmd.extend(args.extra_server_args)

    print("Launching server: " + " ".join(cmd))
    return subprocess.Popen(cmd, env=os.environ.copy(), stdout=sys.stdout, stderr=sys.stderr)


async def amain(args: Args):
    base_url = f"http://{args.server_host}:{args.server_port}/v1"
    print(f"Using server at {base_url}")

    server_proc: Optional[subprocess.Popen] = None
    try:
        if args.launch_server:
            server_proc = launch_sglang_server(args)
            await wait_for_server(f"http://{args.server_host}:{args.server_port}")
        await run_generation(args, base_url)
    finally:
        if server_proc:
            print("Shutting down server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    args = tyro.cli(Args)
    asyncio.run(amain(args))
    print("Done")
