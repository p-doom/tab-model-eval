from src.evaluation import generations
from src.evaluation.prediction_applicators import (
    parse_viewport_command,
    parse_zeta_cursor_position,
)


def test_extract_response_content_zeta_no_codeblock():
    text = "plain response"
    assert generations.extract_response_content(text, generations.InputFormat.ZETA) == text


def test_apply_prediction_sed():
    files = {"a.txt": "one\ntwo"}
    generated = "```bash\nsed -i '2c\\three' a.txt\n```"
    updated, error = generations.apply_prediction(generations.InputFormat.SED, files, generated)
    assert error is None
    assert updated["a.txt"] == "one\nthree"


def test_apply_prediction_sed_non_edit_command():
    files = {"a.txt": "one\ntwo"}
    generated = "```bash\ncat -n a.txt | sed -n '1,2p'\n```"
    updated, error = generations.apply_prediction(generations.InputFormat.SED, files, generated)
    assert error is None
    assert updated == files


def test_parse_viewport_command_sets_cursor():
    cmd = "```bash\ncat -n a.txt | sed -n '10,20p'\n```"
    viewport = parse_viewport_command(cmd)
    assert viewport == {"file_path": "a.txt", "start": 10, "end": 20}


def test_parse_viewport_command_cat_only_defaults():
    cmd = "```bash\ncat -n a.txt\n```"
    viewport = parse_viewport_command(cmd)
    assert viewport == {"file_path": "a.txt", "start": 1, "end": 1}


def test_parse_zeta_cursor_position():
    prediction = (
        "<|editable_region_start|>\n" "line1\n" "li<|user_cursor|>ne2\n" "<|editable_region_end|>"
    )
    cursor = parse_zeta_cursor_position(prediction, {"start": 10, "end": 11}, "a.txt")
    assert cursor == {"file": "a.txt", "line": 13, "column": 2}


def test_apply_prediction_zeta_editable_range():
    files = {"a.txt": "line1\nline2\nline3"}
    generated = "<|editable_region_start|>\nline1\nX\nline3\n<|editable_region_end|>"
    updated, error = generations.apply_prediction(
        generations.InputFormat.ZETA,
        files,
        generated,
        editable_range={"start": 0, "end": 2},
        editable_file="a.txt",
    )
    assert error is None
    assert updated["a.txt"] == "line1\nX\nline3"


def test_build_yaml_result_sets_eval_step_and_metrics():
    raw_result = {
        "task_id": "t1",
        "format": "sed",
        "raw_yaml": {
            "description": "desc",
            "states": [
                {"eval": "NO_EVAL", "files": {"a.txt": "v1"}},
                {"eval": "EVAL", "files": {"a.txt": "v2"}, "judge_assertions": "ok"},
            ],
        },
        "samples": [
            {
                "sample_idx": 0,
                "generated": "g",
                "predicted_files": {"a.txt": "v2"},
                "exact_match": 1,
                "prediction_error": None,
            }
        ],
    }

    result = generations.build_yaml_result(raw_result)
    assert result["metrics"]["num_samples"] == 1
    assert result["metrics"]["num_exact_matches"] == 1
    assert result["metrics"]["pass_at_1"] == 1
    assert result["states"][1]["eval"] == "EVAL"
    assert result["states"][1]["judge_assertions"] == "ok"


def test_filter_by_context_length_skips_long():
    tc = generations.TestCaseWithYaml(
        task_id="t1",
        context=[{"role": "user", "content": "x" * 10000}],
        expected_response="",
        assertions=None,
        input_files={},
        raw_yaml={},
        format=generations.InputFormat.SED,
    )

    valid, skipped = generations.filter_by_context_length(
        [tc],
        system_prompt="system",
        max_context_length=10,
        problem_length=10,
        buffer_tokens=0,
    )

    assert valid == []
    assert skipped == [tc]


def test_build_yaml_result_uses_eval_field():
    raw_result = {
        "task_id": "t2",
        "format": "zeta",
        "raw_yaml": {
            "states": [
                {"eval": "NO_EVAL", "files": {"a.txt": "v1"}},
                {"eval": "EVAL", "files": {"a.txt": "v2"}},
            ],
        },
        "samples": [
            {
                "sample_idx": 0,
                "generated": "g",
                "predicted_files": {"a.txt": "v2"},
                "exact_match": 1,
                "prediction_error": None,
            }
        ],
    }

    result = generations.build_yaml_result(raw_result)
    assert result["format"] == "zeta"
    assert result["states"][1]["eval"] == "EVAL"


def test_compute_cursor_after_text_diff_line_completion():
    cursor = generations.compute_cursor_after_text_diff("value = fo", "value = foo")
    assert cursor == {"line": 1, "column": 11}


def test_compute_cursor_after_text_diff_mid_line_completion():
    cursor = generations.compute_cursor_after_text_diff(
        "result = vari + 1",
        "result = variable + 1",
    )
    assert cursor == {"line": 1, "column": 17}


def test_compute_cursor_after_text_diff_multi_line_edit():
    cursor = generations.compute_cursor_after_text_diff(
        "a=1\nb=2\nc=3",
        "a=1\nb=2\nx=9\ny=10\nc=3",
    )
    assert cursor == {"line": 4, "column": 4}


def test_compute_cursor_after_text_diff_minimal_changed_span():
    cursor = generations.compute_cursor_after_text_diff(
        "import os\nimport sys\nx = 1\n",
        "import os\nimport sys\nx = 2\n",
    )
    assert cursor == {"line": 3, "column": 5}


def test_compute_cursor_after_applied_diff_uses_preferred_file():
    input_files = {
        "pkg/src/main.py": "result = vari + 1",
        "pkg/src/other.py": "a = 1",
    }
    predicted_files = {
        "pkg/src/main.py": "result = variable + 1",
        "pkg/src/other.py": "a = 2",
    }
    cursor = generations.compute_cursor_after_applied_diff(
        input_files,
        predicted_files,
        preferred_file="src/main.py",
    )
    assert cursor == {"file": "pkg/src/main.py", "line": 1, "column": 17}


def test_compute_cursor_after_applied_diff_no_change_returns_none():
    files = {"a.txt": "same"}
    cursor = generations.compute_cursor_after_applied_diff(files, {"a.txt": "same"})
    assert cursor is None
