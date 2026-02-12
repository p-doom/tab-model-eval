import asyncio
import glob
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
import tyro
import yaml
import wandb

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from src.applicators.prediction_applicators import (
    apply_sed_prediction,
    apply_zeta_prediction_to_editable,
    parse_viewport_command,
    parse_zeta_cursor_position,
)
from src.utils.types import (
    InputFormat,
    TestCaseWithYaml,
)

from src.formats.base import FormatConverter
from src.formats.sed import SEDConverter
from src.formats.zeta import ZetaConverter


@dataclass
class Args:
    wandb_project: str = "tab-model-eval"
    wandb_name: str = "metrics_eval"
    wandb_eval_type: str = "metrics_eval"
    wandb_tags: list[str] = field(default_factory=list)
    wandb_id: Optional[str] = None
    wandb_group: str = "evals"

    input_format: str = ""
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


def load_yaml_files(yaml_dir: str, limit: int = -1) -> List[Dict[str, Any]]:
    pattern = os.path.join(yaml_dir, "*.yaml")
    yaml_files = sorted(glob.glob(pattern))

    if limit > 0:
        yaml_files = yaml_files[:limit]

    results = []
    for path in yaml_files:
        with open(path, "r") as f:
            results.append(yaml.safe_load(f))

    return results


_CONVERTERS: Dict[InputFormat, FormatConverter] = {
    InputFormat.SED: SEDConverter(),
    InputFormat.ZETA: ZetaConverter(),
}


def convert_yaml_to_test_cases(
    yaml_data_list: List[Dict[str, Any]],
    input_format: InputFormat,
) -> List[TestCaseWithYaml]:
    converter = _CONVERTERS.get(input_format)
    if converter is None:
        raise ValueError(f"Unsupported format: {input_format.value}")
    return converter.convert(yaml_data_list)


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
        match = re.search(r"`````\n(.*?)\n`````", response_text, re.DOTALL)
        if match:
            return match.group(1)
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
    if format_type == InputFormat.SED:
        return apply_sed_prediction(input_files, generated)
    elif format_type == InputFormat.ZETA:
        return apply_zeta_prediction_to_editable(
            input_files, generated, editable_range, editable_file
        )
    return None, f"Unsupported format: {format_type.value}"


async def generate_single_sample(
    client: AsyncOpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    presence_penalty: float,
    max_tokens: int,
    max_attempts: int,
    task_id: str,
) -> Optional[str]:
    delay = 0.25
    for attempt in range(max_attempts):
        try:
            resp = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                presence_penalty=presence_penalty,
                n=1,  # Always use n=1 to avoid SGLang bug with n>1 on some models
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            if resp.choices:
                return resp.choices[0].message.content or ""
            return ""
        except Exception as e:
            print(f"Error on {task_id} sample (attempt {attempt + 1}): {e}")
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay)
                delay *= 2
    return None


async def generate_for_test_case(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    system_prompt: str,
    test_case: TestCaseWithYaml,
    args: Args,
) -> Dict[str, Any]:
    if test_case.unsupported:
        return {
            "task_id": test_case.task_id,
            "format": test_case.format.value,
            "input_files": test_case.input_files,
            "raw_yaml": test_case.raw_yaml,
            "expected_response": test_case.expected_response,
            "assertions": test_case.assertions,
            "samples": [],
            "error": f"unsupported_test_case: {test_case.format.value} format does not support this test case type",
        }

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(test_case.context)

    async with sem:
        # Generate sequentially: n>1 is broken in SGLang for some models.
        samples = []
        for idx in range(args.num_samples):
            response_text = await generate_single_sample(
                client=client,
                model_name=args.model_name,
                messages=messages,
                presence_penalty=args.presence_penalty,
                max_tokens=args.max_new_tokens,
                max_attempts=args.max_attempts,
                task_id=test_case.task_id,
            )

            if response_text is None:
                return {
                    "task_id": test_case.task_id,
                    "format": test_case.format.value,
                    "input_files": test_case.input_files,
                    "raw_yaml": test_case.raw_yaml,
                    "expected_response": test_case.expected_response,
                    "assertions": test_case.assertions,
                    "samples": samples,
                    "error": f"Failed to generate sample {idx} after {args.max_attempts} attempts",
                }

            generated = extract_response_content(response_text, test_case.format)
            response_match = int(generated == test_case.expected_response)
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

            cursor_match = int(predicted_cursor == test_case.expected_cursor)
            exact_match = int(response_match == 1 and cursor_match == 1)

            samples.append(
                {
                    "sample_idx": idx,
                    "response_text": response_text,
                    "generated": generated,
                    "response_match": response_match,
                    "cursor_match": cursor_match,
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


def build_yaml_result(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    task_id = raw_result["task_id"]
    format_type = raw_result["format"]
    raw_yaml = raw_result.get("raw_yaml", {})
    samples_raw = raw_result.get("samples", [])

    yaml_states = raw_yaml.get("states", [])

    eval_step = 0
    for i, state in enumerate(yaml_states):
        eval = state.get("eval", "NO_EVAL")
        if eval == "EVAL":
            eval_step = i
            break

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

    samples = []
    for s in samples_raw:
        samples.append(
            {
                "sample_idx": s["sample_idx"],
                "step": eval_step,
                "predicted_files": s.get("predicted_files"),
                "predicted_cursor": s.get("predicted_cursor"),
                "predicted_raw": s.get("generated", ""),
                "response_match": s.get("response_match", 0),
                "cursor_match": s.get("cursor_match", 0),
                "exact_match": s.get("exact_match", 0),
                "prediction_error": s.get("prediction_error"),
            }
        )

    task_num_samples = len(samples)
    task_num_errors = sum(1 for s in samples if s.get("prediction_error") is not None)

    task_response_matches = sum(1 for s in samples if s.get("response_match", 0) == 1)
    task_cursor_matches = sum(1 for s in samples if s.get("cursor_match", 0) == 1)
    task_exact_matches = sum(1 for s in samples if s.get("exact_match", 0) == 1)

    task_response_pass_at_k = 1 if task_response_matches > 0 else 0
    task_cursor_pass_at_k = 1 if task_cursor_matches > 0 else 0
    task_exact_pass_at_k = 1 if task_exact_matches == 1 else 0

    return {
        "task_id": task_id,
        "format": format_type,
        "description": raw_yaml.get("description", ""),
        "states": states,
        "samples": samples,
        "metrics": {
            "task_num_samples": task_num_samples,
            "task_num_errors": task_num_errors,
            "task_response_matches": task_response_matches,
            "task_cursor_matches": task_cursor_matches,
            "task_exact_matches": task_exact_matches,
            "task_response_pass_at_k": task_response_pass_at_k,
            "task_cursor_pass_at_k": task_cursor_pass_at_k,
            "task_exact_pass_at_k": task_exact_pass_at_k,
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
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    output = {
        "config": config,
        "results": results,
    }

    with open(output_file, "w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


async def run_generation(args: Args, base_url: str, wandb_run: Optional[wandb.Run] = None):
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

    total_num_samples = sum(r["metrics"]["task_num_samples"] for r in yaml_results)
    num_samples_per_task = total_num_samples / len(yaml_results)
    total_num_errors = sum(r["metrics"]["task_num_errors"] for r in yaml_results)

    total_response_matches = sum(r["metrics"]["task_response_matches"] for r in yaml_results)
    total_cursor_matches = sum(r["metrics"]["task_cursor_matches"] for r in yaml_results)
    total_exact_matches = sum(r["metrics"]["task_exact_matches"] for r in yaml_results)

    response_pass_at_k = sum(r["metrics"]["task_response_pass_at_k"] for r in yaml_results)
    cursor_pass_at_k = sum(r["metrics"]["task_cursor_pass_at_k"] for r in yaml_results)
    exact_pass_at_k = sum(r["metrics"]["task_exact_pass_at_k"] for r in yaml_results)

    if wandb_run is not None:
        wandb_run.log(
            {
                f"{args.wandb_eval_type}/total_num_samples": total_num_samples,
                f"{args.wandb_eval_type}/num_samples_per_task": num_samples_per_task,
                f"{args.wandb_eval_type}/total_num_errors": total_num_errors,
                f"{args.wandb_eval_type}/total_response_matches": total_response_matches,
                f"{args.wandb_eval_type}/total_cursor_matches": total_cursor_matches,
                f"{args.wandb_eval_type}/total_exact_matches": total_exact_matches,
                f"{args.wandb_eval_type}/response_pass_at_k": response_pass_at_k,
                f"{args.wandb_eval_type}/cursor_pass_at_k": cursor_pass_at_k,
                f"{args.wandb_eval_type}/exact_pass_at_k": exact_pass_at_k,
            }
        )

    print("\n" + "=" * 50)
    print("Generation Complete")
    print("=" * 50)
    print(f"Tasks: {len(yaml_results)}")
    print(f"Total samples: {total_num_samples}, samples per task: {num_samples_per_task}")
    print(f"Errors: {total_num_errors}, {total_num_errors / total_num_samples * 100:.1f}%")
    print(
        f"Response matches: {total_response_matches}, {total_response_matches / total_num_samples * 100:.1f}%"
    )
    print(
        f"Cursor matches: {total_cursor_matches}, {total_cursor_matches / total_num_samples * 100:.1f}%"
    )
    print(
        f"Exact matches: {total_exact_matches}, {total_exact_matches / total_num_samples * 100:.1f}%"
    )
    print(
        f"Response pass@k: {response_pass_at_k}, {response_pass_at_k / len(yaml_results) * 100:.1f}%"
    )
    print(f"Cursor pass@k: {cursor_pass_at_k}, {cursor_pass_at_k / len(yaml_results) * 100:.1f}%")
    print(f"Exact pass@k: {exact_pass_at_k}, {exact_pass_at_k / len(yaml_results) * 100:.1f}%")


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


async def amain(args: Args, wandb_run: Optional[wandb.Run] = None):
    base_url = f"http://{args.server_host}:{args.server_port}/v1"
    print(f"Using server at {base_url}")

    server_proc: Optional[subprocess.Popen] = None
    try:
        if args.launch_server:
            server_proc = launch_sglang_server(args)
            await wait_for_server(f"http://{args.server_host}:{args.server_port}")
        await run_generation(args, base_url, wandb_run)
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

    wandb_run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        id=args.wandb_id,
        resume="allow" if args.wandb_id else None,
        group=args.wandb_group,
        tags=args.wandb_tags,
        config={"eval_type": args.wandb_eval_type},
    )

    asyncio.run(amain(args, wandb_run))

    if wandb_run is not None:
        wandb_run.finish()

    print("\nDone")
