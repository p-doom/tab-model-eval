import asyncio
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
import tyro
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from .format_utils import InputFormat, UnifiedTestCase, load_test_cases, get_format_stats
from .prediction_applicators import apply_prediction, extract_expected_files
from .yaml_output import (
    build_state,
    build_sample_prediction,
    build_generation_result,
    write_yaml_output,
)


@dataclass
class Args:
    test_cases_file: str = "data/eval/handcrafted_test_cases/handcrafted_test_cases.jsonl"
    output_file: str = "data/eval/generations/generations.yaml"
    limit: int = -1

    system_prompt_file: str = "data/prompts/generation_system_prompt_v2.md"
    zeta_system_prompt_file: str = "data/prompts/zeta_system_prompt.md"
    viewport_radius: int = 10

    model_name: str = "default"
    model_path: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    presence_penalty: float = 1.5
    num_samples: int = 5
    max_new_tokens: int = 5000

    server_host: str = "0.0.0.0"
    server_port: int = 30000
    context_length: int = 40960
    problem_length: int = 40960
    mem_fraction_static: float = 0.95
    api_key: str = "EMPTY"
    tp_size: int = 1
    lora_paths: Optional[List[str]] = None
    launch_server: bool = True
    extra_server_args: Optional[List[str]] = None

    concurrency: int = 16
    max_connections: int = 256
    keepalive: int = 60
    max_attempts: int = 6
    timeout: float = 300.0


def estimate_token_count(messages: List[Dict[str, str]]) -> int:
    total_chars = sum(len(msg.get("content", "")) for msg in messages)
    return total_chars // 3


def filter_by_context_length(
    test_cases: List[UnifiedTestCase],
    system_prompt: str,
    max_context_length: int,
    problem_length: int,
    buffer_tokens: int = 512,
) -> tuple[List[UnifiedTestCase], List[UnifiedTestCase]]:
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
        if "<output>" in response_text:
            match = re.search(r"(<output>.*?</output>)", response_text, re.DOTALL)
            return match.group(1) if match else response_text
        return response_text
    return response_text


async def generate_for_test_case(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    system_prompt: str,
    test_case: UnifiedTestCase,
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
                        test_case.format, test_case.input_files, generated
                    )

                    samples.append(
                        {
                            "sample_idx": idx,
                            "response_text": response_text,
                            "generated": generated,
                            "exact_match": exact_match,
                            "predicted_files": predicted_files if error is None else None,
                            "prediction_error": error,
                        }
                    )

                return {
                    "task_id": test_case.task_id,
                    "format": test_case.format.value,
                    "input_files": test_case.input_files,
                    "expected_response": test_case.expected_response,
                    "assertions": test_case.assertions,
                    "cursor": test_case.cursor,
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
                        "expected_response": test_case.expected_response,
                        "assertions": test_case.assertions,
                        "cursor": test_case.cursor,
                        "samples": [],
                        "error": str(e),
                    }
                await asyncio.sleep(delay)
                delay *= 2

    return {"task_id": test_case.task_id, "error": "max_attempts_exceeded", "samples": []}


def build_yaml_result(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    task_id = raw_result["task_id"]
    format_type = raw_result["format"]
    input_files = raw_result.get("input_files", {})
    expected_response = raw_result.get("expected_response", "")
    assertions = raw_result.get("assertions")
    cursor = raw_result.get("cursor")
    samples_raw = raw_result.get("samples", [])

    states = []
    states.append(build_state(step=0, eval_tag="NO_EVAL", files=input_files, cursor=cursor))

    format_enum = InputFormat(format_type) if format_type else InputFormat.UNKNOWN
    expected_files = extract_expected_files(format_enum, expected_response, input_files)
    states.append(
        build_state(step=1, eval_tag="EVAL", files=expected_files, judge_assertions=assertions)
    )

    samples = []
    for s in samples_raw:
        samples.append(
            build_sample_prediction(
                sample_idx=s["sample_idx"],
                step=1,
                predicted_files=s.get("predicted_files"),
                predicted_raw=s.get("generated", ""),
                exact_match=s.get("exact_match", 0),
                prediction_error=s.get("prediction_error"),
            )
        )

    return build_generation_result(
        task_id=task_id, format_type=format_type, states=states, samples=samples
    )


async def run_generation(args: Args, base_url: str):
    print(f"Loading test cases from {args.test_cases_file}...")
    test_cases = load_test_cases(args.test_cases_file, limit=args.limit)

    stats = get_format_stats(test_cases)
    print(f"Loaded {stats['total']} test cases (SED: {stats['sed']}, Zeta: {stats['zeta']})")

    if stats["zeta"] > stats["sed"]:
        prompt_file = args.zeta_system_prompt_file
        if not os.path.exists(prompt_file):
            prompt_file = args.system_prompt_file
    else:
        prompt_file = args.system_prompt_file

    print(f"Using system prompt: {prompt_file}")
    with open(prompt_file, "r") as f:
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

    total_samples = sum(r.get("metrics", {}).get("num_samples", 0) for r in yaml_results)
    total_exact = sum(r.get("metrics", {}).get("num_exact_matches", 0) for r in yaml_results)
    total_pass = sum(r.get("metrics", {}).get("pass_at_1", 0) for r in yaml_results)

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
    ]

    if args.lora_paths:
        cmd.append("--lora-paths")
        cmd.extend(args.lora_paths)

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
