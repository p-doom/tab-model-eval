"""
Shared utilities for evaluation scripts.

This module contains common utilities used by both sglang_eval_metrics.py
(format validation, deterministic metrics) and sglang_eval_judge.py
(LLM-as-a-judge evaluation).
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Local logger for offline mode
# ----------------------------
class LocalLogger:
    """A simple local logger that saves metrics to JSON files for later sync to wandb."""

    def __init__(
        self,
        log_dir: str,
        run_id: str,
        run_name: str,
        project: str,
        config: Optional[dict] = None,
        tags: Optional[list] = None,
    ):
        self.log_dir = os.path.join(log_dir, run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        self.run_id = run_id
        self.run_name = run_name
        self.project = project
        self.config = config or {}
        self.tags = tags or []
        self.metrics_file = os.path.join(self.log_dir, "metrics.jsonl")

        # Save run metadata
        metadata_file = os.path.join(self.log_dir, "metadata.json")
        if os.path.exists(metadata_file):
            print(
                f"Metadata file already exists for run_id={run_id} at {metadata_file}. "
                f"Existing metadata will be reused."
            )
        else:
            with open(metadata_file, "w") as f:
                json.dump(
                    {
                        "run_id": run_id,
                        "run_name": run_name,
                        "project": project,
                        "config": config,
                        "tags": tags,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                    f,
                    indent=2,
                )
        print(f"LocalLogger initialized. Logs will be saved to: {self.log_dir}")

    def log(self, metrics: dict):
        """Append metrics to the JSONL file."""
        metrics_with_timestamp = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **metrics}
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics_with_timestamp) + "\n")
        print(f"Logged metrics to {self.metrics_file}: eval_step={metrics.get('eval_step', 'N/A')}")

    def finish(self):
        """Called when logging is complete."""
        print(f"LocalLogger finished. All logs saved to: {self.log_dir}")


# ----------------------------
# Dataset helpers
# ----------------------------
def load_dataset(filepath: str) -> Dict[str, Any]:
    """Load a JSON dataset file."""
    with open(filepath, "r") as f:
        return json.loads(f.read())


def save_dataset(filepath: str, data: Dict[str, Any]) -> None:
    """Save a dataset to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def estimate_token_count(messages: List[Dict[str, str]]) -> int:
    """
    Rough estimate of token count for a list of messages.
    Assumes ~3 characters per token as a conservative estimate.
    """
    total_chars = sum(len(msg.get("content", "")) for msg in messages)
    return total_chars // 3


def filter_tasks_by_context_length(
    test_cases: List[Dict[str, Any]],
    system_prompt: str,
    prompt_template: str = "",
    max_context_length: int = 40960,
    problem_length: int = 40960,
    buffer_tokens: int = 512,
    include_context: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter out test cases whose context would exceed the model's context length.
    Returns (valid_cases, skipped_cases)
    """
    valid_cases = []
    skipped_cases = []

    for tc in test_cases:
        messages = [{"role": "system", "content": system_prompt}]
        if include_context:
            messages.extend(tc.get("context", []))
        if prompt_template:
            messages.append({"role": "user", "content": prompt_template})
        estimated_tokens = estimate_token_count(messages)

        length = estimated_tokens + buffer_tokens
        if length <= max_context_length and length <= problem_length:
            valid_cases.append(tc)
        else:
            print(
                f"Skipping {tc['task_id']}: estimated {estimated_tokens} tokens "
                f"(limit: {max_context_length}, problem_length: {problem_length})"
            )
            skipped_cases.append(
                {
                    "task_id": tc["task_id"],
                    "estimated_tokens": estimated_tokens,
                    "reason": "context_too_long",
                }
            )

    return valid_cases, skipped_cases


# ----------------------------
# Command format validation
# ----------------------------
def check_command_format(command: str, expected_command: str) -> Tuple[bool, str]:
    r"""
    Validates that a command follows the required edit format.

    Only applies edit-specific validation when expected_command is a sed edit command.

    Returns (is_valid, reason):
    - If expected is not a sed edit: always valid, returns (True, "non_edit_task")
    - If expected is a sed edit, the generated command must:
      - Use one of the 4 allowed sed patterns (not s/old/new/ substitution)
      - Not use file redirection (>, >>, tee) to modify files

    Valid sed patterns:
      1. Replace block: sed -i 'START,ENDc\...'
      2. Delete block: sed -i 'START,ENDd'
      3. Insert before: sed -i 'STARTi\...'
      4. Append to end: sed -i '$a\...'
    """
    if "sed -i" not in expected_command:
        return (True, "non_edit_task")

    # Check for file redirection patterns (>, >>, tee) - these are not allowed for edits
    file_redirection_pattern = r"(^|[;&|])\s*(echo|cat|printf)\s+.*\s*>{1,2}\s*\S+"
    if re.search(file_redirection_pattern, command):
        return (False, "invalid_file_redirection")

    # Check for tee command writing to files
    tee_pattern = r"(^|[;&|])\s*tee\s+"
    if re.search(tee_pattern, command):
        return (False, "invalid_tee_redirection")

    # Check for heredoc patterns: cat << EOF > file
    heredoc_pattern = r"<<\s*\w+.*>"
    if re.search(heredoc_pattern, command):
        return (False, "invalid_heredoc_redirection")

    # If generated command doesn't use sed -i at all, it's invalid for an edit task
    if "sed -i" not in command:
        return (False, "missing_sed_edit")

    # Check for forbidden s/old/new/ substitution pattern
    substitution_pattern = r"sed\s+-i\s+['\"][^'\"]*\d*s/"
    if re.search(substitution_pattern, command):
        return (False, "invalid_sed_substitution")

    # Allowed edit patterns:
    # 1. Replace block: sed -i 'START,ENDc\...' or sed -i 'LINEc\...' (single line)
    # 2. Delete block: sed -i 'START,ENDd' or sed -i 'LINEd'
    # 3. Insert before: sed -i 'STARTi\...'
    # 4. Append to end: sed -i '$a\...'

    # Replace: accept both range (5,10c\) and single line (5c\)
    replace_pattern = r"sed\s+-i\s+['\"](\d+)(,\d+)?c\\"
    # Delete: accept both range and single line
    delete_pattern = r"sed\s+-i\s+['\"](\d+)(,\d+)?d['\"]"
    insert_pattern = r"sed\s+-i\s+['\"](\d+)i\\"
    append_pattern = r"sed\s+-i\s+['\"]\$a\\"

    if (
        re.search(replace_pattern, command)
        or re.search(delete_pattern, command)
        or re.search(insert_pattern, command)
        or re.search(append_pattern, command)
    ):
        return (True, "valid_edit_command")

    return (False, "invalid_format")


# ----------------------------
# HumanEval Sandbox Execution
# ----------------------------
def is_humaneval_task(task_id: str) -> bool:
    """Check if a task is a HumanEval task based on task_id prefix."""
    return task_id.startswith("humaneval_")


def extract_bash_command(text: str) -> Optional[str]:
    """Extract the bash command from a markdown code block."""
    match = re.search(r"```bash\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def parse_file_content_from_context(context: List[Dict[str, Any]]) -> Tuple[str, str, List[str]]:
    """
    Parse the file path and content from the context.
    Returns (file_path, file_content_without_line_numbers, lines).

    The context contains stdout with numbered lines like:
         2\t
         3\tdef foo():
    """
    # Find the last user message with stdout containing numbered file content
    file_path = None
    file_lines = []

    for msg in reversed(context):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if "<stdout>" in content:
                # Extract the stdout content
                match = re.search(r"<stdout>\s*(.*?)\s*</stdout>", content, re.DOTALL)
                if match:
                    stdout_content = match.group(1)
                    # Parse numbered lines (format: "     N\tCONTENT")
                    lines = stdout_content.split("\n")
                    for line in lines:
                        # Match line number format: spaces + number + tab + content
                        line_match = re.match(r"\s*(\d+)\t(.*)", line)
                        if line_match:
                            file_lines.append(line_match.group(2))
                    break

    # Extract file path from context (from assistant's sed/cat command)
    for msg in context:
        if msg.get("role") == "assistant":
            cmd = msg.get("content", "")
            # Look for file path in sed or cat commands
            path_match = re.search(r"(?:sed -i|cat -n|cat)\s+['\"]?([^\s'\"]+\.py)", cmd)
            if path_match:
                file_path = path_match.group(1)
                break

    file_content = "\n".join(file_lines)
    return file_path or "", file_content, file_lines


def validate_sed_command_for_sandbox(
    command: str,
    sandbox_dir: str,
    allowed_paths: List[str],
) -> Tuple[bool, str, Optional[str]]:
    """
    Validate that a sed command is safe to execute in the sandbox.

    Returns (is_safe, reason, rewritten_command).
    The rewritten command has paths adjusted for the sandbox.
    """
    # Extract the bash command if wrapped in markdown
    bash_cmd = extract_bash_command(command)
    if not bash_cmd:
        bash_cmd = command

    # Must be a sed -i command
    if "sed -i" not in bash_cmd:
        return (False, "not_sed_command", None)

    # Check for dangerous patterns
    dangerous_patterns = [
        r";\s*rm\s",
        r";\s*mv\s",
        r";\s*cp\s",
        r";\s*chmod\s",
        r";\s*chown\s",
        r"\$\(",  # Command substitution
        r"`",  # Backtick command substitution
        r">\s*/",  # Redirect to absolute path
        r";\s*python",
        r";\s*bash",
        r";\s*sh\s",
        r";\s*eval\s",
        r";\s*exec\s",
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, bash_cmd):
            return (False, f"dangerous_pattern_{pattern}", None)

    # Extract and validate file path from sed command
    # Pattern: sed -i '...' /path/to/file
    # Use DOTALL to handle multi-line sed commands with backslash continuation
    sed_path_match = re.search(r"sed\s+-i\s+['\"].*?['\"]\s+([^\s&|;]+)", bash_cmd, re.DOTALL)
    if not sed_path_match:
        return (False, "cannot_extract_path", None)

    original_path = sed_path_match.group(1)

    # Check if path is in allowed paths
    path_allowed = False
    for allowed in allowed_paths:
        if allowed in original_path:
            path_allowed = True
            break

    if not path_allowed:
        return (False, f"path_not_allowed_{original_path}", None)

    # Rewrite path to sandbox
    # Extract just the filename
    filename = os.path.basename(original_path)
    sandbox_path = os.path.join(sandbox_dir, filename)

    # Rewrite the command
    # Only rewrite the sed part, ignore the cat part after &&
    # Use DOTALL to handle multi-line sed commands
    sed_part_match = re.match(r"(sed\s+-i\s+['\"].*?['\"])\s+[^\s&|;]+", bash_cmd, re.DOTALL)
    if sed_part_match:
        sed_part = sed_part_match.group(1)
        rewritten_cmd = f"{sed_part} {sandbox_path}"
        return (True, "valid", rewritten_cmd)

    return (False, "cannot_rewrite", None)


def create_humaneval_sandbox(
    file_path: str,
    file_content: str,
    base_sandbox_dir: Optional[str] = None,
) -> str:
    """
    Create a sandbox directory with the initial file state.
    Returns the path to the sandbox directory.
    """
    if base_sandbox_dir:
        sandbox_dir = tempfile.mkdtemp(dir=base_sandbox_dir)
    else:
        sandbox_dir = tempfile.mkdtemp(prefix="humaneval_sandbox_")

    # Create the file in the sandbox
    filename = os.path.basename(file_path)
    sandbox_file = os.path.join(sandbox_dir, filename)

    with open(sandbox_file, "w") as f:
        f.write(file_content)

    return sandbox_dir


def execute_sed_in_sandbox(
    sed_command: str,
    sandbox_dir: str,
    timeout: float = 5.0,
) -> Tuple[bool, str, str]:
    """
    Execute a sed command in the sandbox.
    Returns (success, stdout, stderr).
    """
    try:
        result = subprocess.run(
            sed_command,
            shell=True,
            cwd=sandbox_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return (result.returncode == 0, result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return (False, "", "timeout")
    except Exception as e:
        return (False, "", str(e))


def run_humaneval_test(
    sandbox_dir: str,
    entry_point: str,
    test_code: str,
    timeout: float = 10.0,
) -> Tuple[bool, str]:
    """
    Run HumanEval test on the file in the sandbox.
    Returns (passed, error_message).
    """
    filename = f"{entry_point}.py"
    sandbox_file = os.path.join(sandbox_dir, filename)

    if not os.path.exists(sandbox_file):
        return (False, f"file_not_found_{filename}")

    # Read the modified file
    with open(sandbox_file, "r") as f:
        code = f.read()

    # Combine code with test
    full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"

    # Write test file
    test_file = os.path.join(sandbox_dir, "test_runner.py")
    with open(test_file, "w") as f:
        f.write(full_code)

    # Run the test
    try:
        result = subprocess.run(
            ["python", test_file],
            cwd=sandbox_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return (True, "")
        else:
            error = result.stderr or result.stdout
            # Truncate long errors
            if len(error) > 500:
                error = error[:500] + "..."
            return (False, error)
    except subprocess.TimeoutExpired:
        return (False, "timeout")
    except Exception as e:
        return (False, str(e))


def cleanup_sandbox(sandbox_dir: str) -> None:
    """Remove the sandbox directory."""
    try:
        shutil.rmtree(sandbox_dir)
    except Exception:
        pass  # Ignore cleanup errors


def reconstruct_file_from_context(
    context: List[Dict[str, Any]],
    humaneval_meta: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Reconstruct the initial file state from context and humaneval_meta.
    This handles the case where context only shows a viewport of the file.

    Returns (file_path, full_file_content).
    """
    # First try to parse from context
    file_path, content_from_context, lines = parse_file_content_from_context(context)

    # Get entry point for filename
    entry_point = humaneval_meta.get("entry_point", "")
    if not file_path and entry_point:
        file_path = f"{entry_point}.py"

    # If we got content from context, use it
    if lines:
        return file_path, "\n".join(lines)

    return file_path, ""


def evaluate_humaneval_sample(
    generated_command: str,
    context: List[Dict[str, Any]],
    humaneval_meta: Dict[str, Any],
    base_sandbox_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single HumanEval sample by executing in sandbox.

    Returns dict with:
    - execution_success: bool
    - test_passed: bool
    - error: str (if any)
    - sandbox_validated: bool
    """
    result = {
        "execution_success": False,
        "test_passed": False,
        "error": "",
        "sandbox_validated": False,
    }

    # Reconstruct file from context
    file_path, file_content = reconstruct_file_from_context(context, humaneval_meta)

    if not file_content:
        result["error"] = "cannot_reconstruct_file"
        return result

    entry_point = humaneval_meta.get("entry_point", "")
    if not entry_point:
        result["error"] = "missing_entry_point"
        return result

    # Create sandbox
    sandbox_dir = create_humaneval_sandbox(
        file_path=f"{entry_point}.py",
        file_content=file_content,
        base_sandbox_dir=base_sandbox_dir,
    )

    try:
        # Validate and rewrite sed command for sandbox
        allowed_paths = ["/home/user/projects/", entry_point]
        is_safe, reason, rewritten_cmd = validate_sed_command_for_sandbox(
            generated_command,
            sandbox_dir,
            allowed_paths,
        )

        if not is_safe:
            result["error"] = f"sandbox_validation_failed_{reason}"
            return result

        result["sandbox_validated"] = True

        # Execute sed command (rewritten_cmd is guaranteed to be str when is_safe is True)
        assert rewritten_cmd is not None
        success, stdout, stderr = execute_sed_in_sandbox(rewritten_cmd, sandbox_dir)
        if not success:
            result["error"] = f"sed_execution_failed_{stderr}"
            return result

        result["execution_success"] = True

        # Run HumanEval test
        test_code = humaneval_meta.get("test", "")
        if not test_code:
            result["error"] = "missing_test_code"
            return result

        passed, test_error = run_humaneval_test(sandbox_dir, entry_point, test_code)
        result["test_passed"] = passed
        if not passed:
            result["error"] = f"test_failed_{test_error}"

        return result

    finally:
        cleanup_sandbox(sandbox_dir)
