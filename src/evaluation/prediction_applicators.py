import re
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from .format_utils import InputFormat


def extract_bash_command(text: str) -> str:
    match = re.search(r"```bash\s*\n(.*?)\n```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def parse_sed_command(sed_command: str) -> Optional[Dict[str, Any]]:
    sed_part = sed_command.split("&&")[0].strip()
    sed_part = extract_bash_command(sed_part) if "```" in sed_part else sed_part

    if "sed -i" not in sed_part:
        return None

    # Delete: sed -i 'Nd' or sed -i 'N,Md'
    delete_match = re.search(r"sed\s+-i\s+['\"](\d+)(?:,(\d+))?d['\"]\s+([^\s&|;]+)", sed_part)
    if delete_match:
        start = int(delete_match.group(1))
        end = int(delete_match.group(2)) if delete_match.group(2) else start
        return {
            "operation": "delete",
            "start_line": start,
            "end_line": end,
            "content": None,
            "file_path": delete_match.group(3),
        }

    # Replace: sed -i 'Nc\text' or sed -i 'N,Mc\text'
    replace_match = re.search(
        r"sed\s+-i\s+'(\d+)(?:,(\d+))?c\\?(.*?)'\s+([^\s&|;]+)", sed_part, re.DOTALL
    )
    if not replace_match:
        replace_match = re.search(
            r'sed\s+-i\s+"(\d+)(?:,(\d+))?c\\?(.*?)"\s+([^\s&|;]+)', sed_part, re.DOTALL
        )
    if replace_match:
        start = int(replace_match.group(1))
        end = int(replace_match.group(2)) if replace_match.group(2) else start
        content = re.sub(r"\\\n", "\n", replace_match.group(3))
        return {
            "operation": "replace",
            "start_line": start,
            "end_line": end,
            "content": content,
            "file_path": replace_match.group(4),
        }

    # Insert: sed -i 'Ni\text'
    insert_match = re.search(r"sed\s+-i\s+'(\d+)i\\?(.*?)'\s+([^\s&|;]+)", sed_part, re.DOTALL)
    if not insert_match:
        insert_match = re.search(r'sed\s+-i\s+"(\d+)i\\?(.*?)"\s+([^\s&|;]+)', sed_part, re.DOTALL)
    if insert_match:
        line = int(insert_match.group(1))
        content = re.sub(r"\\\n", "\n", insert_match.group(2))
        return {
            "operation": "insert",
            "start_line": line,
            "end_line": line,
            "content": content,
            "file_path": insert_match.group(3),
        }

    # Append: sed -i '$a\text'
    append_match = re.search(r"sed\s+-i\s+'\$a\\?(.*?)'\s+([^\s&|;]+)", sed_part, re.DOTALL)
    if not append_match:
        append_match = re.search(r'sed\s+-i\s+"\$a\\?(.*?)"\s+([^\s&|;]+)', sed_part, re.DOTALL)
    if append_match:
        content = re.sub(r"\\\n", "\n", append_match.group(1))
        return {
            "operation": "append",
            "start_line": -1,
            "end_line": -1,
            "content": content,
            "file_path": append_match.group(2),
        }

    return None


def apply_sed_to_content(content: str, sed_parsed: Dict[str, Any]) -> str:
    lines = content.split("\n")
    op = sed_parsed["operation"]
    start = sed_parsed["start_line"]
    end = sed_parsed["end_line"]
    new_content = sed_parsed.get("content")

    if op == "delete":
        del lines[start - 1 : end]
    elif op == "replace":
        replacement_lines = new_content.split("\n") if new_content else []
        lines[start - 1 : end] = replacement_lines
    elif op == "insert":
        insert_lines = new_content.split("\n") if new_content else []
        for i, line in enumerate(insert_lines):
            lines.insert(start - 1 + i, line)
    elif op == "append":
        append_lines = new_content.split("\n") if new_content else []
        lines.extend(append_lines)

    return "\n".join(lines)


def _find_matching_path(files: Dict[str, str], target: str) -> Optional[str]:
    for path in files:
        if path == target or path.endswith(target) or target.endswith(path):
            return path
    return None


def apply_sed_prediction(
    files: Dict[str, str],
    prediction: str,
) -> Tuple[Dict[str, str], Optional[str]]:
    command = extract_bash_command(prediction)
    parsed = parse_sed_command(command)
    if parsed is None:
        return files, f"failed_to_parse_sed: {command[:100]}"

    matched_path = _find_matching_path(files, parsed["file_path"])
    if matched_path is None:
        return files, f"file_not_found: {parsed['file_path']}"

    updated_files = deepcopy(files)
    try:
        updated_files[matched_path] = apply_sed_to_content(files[matched_path], parsed)
        return updated_files, None
    except Exception as e:
        return files, f"apply_error: {e}"


def parse_zeta_output(output: str) -> Optional[Dict[str, str]]:
    files = {}

    output_match = re.search(r"<output>\s*(.*?)\s*</output>", output, re.DOTALL)
    output_content = output_match.group(1) if output_match else output

    code_blocks = re.findall(r"```([^\n]+)\n(.*?)```", output_content, re.DOTALL)

    for file_path, content in code_blocks:
        file_path = file_path.strip()
        content = re.sub(r"<\|editable_region_start\|>\n?", "", content)
        content = re.sub(r"<\|editable_region_end\|>\n?", "", content)
        content = re.sub(r"<\|user_cursor_is_here\|>", "", content)
        files[file_path] = content

    return files if files else None


def apply_zeta_prediction(
    files: Dict[str, str],
    prediction: str,
) -> Tuple[Dict[str, str], Optional[str]]:
    parsed = parse_zeta_output(prediction)
    if parsed is None:
        return files, "failed_to_parse_zeta_output"

    updated_files = deepcopy(files)

    for file_path, new_content in parsed.items():
        matched_path = _find_matching_path(files, file_path)
        if matched_path:
            updated_files[matched_path] = new_content
        else:
            updated_files[file_path] = new_content

    return updated_files, None


def apply_prediction(
    format_type: InputFormat,
    files: Dict[str, str],
    prediction: str,
) -> Tuple[Dict[str, str], Optional[str]]:
    if format_type == InputFormat.SED:
        return apply_sed_prediction(files, prediction)
    elif format_type == InputFormat.ZETA:
        return apply_zeta_prediction(files, prediction)
    else:
        return files, f"unsupported_format: {format_type}"


def extract_expected_files(
    format_type: InputFormat,
    expected_response: str,
    input_files: Dict[str, str],
) -> Dict[str, str]:
    if format_type == InputFormat.ZETA:
        parsed = parse_zeta_output(expected_response)
        if parsed:
            result = deepcopy(input_files)
            for path, content in parsed.items():
                matched = _find_matching_path(result, path)
                if matched:
                    result[matched] = content
                else:
                    result[path] = content
            return result

    elif format_type == InputFormat.SED:
        result, _ = apply_sed_prediction(input_files, expected_response)
        return result

    return input_files
