import json
from pathlib import Path

from src.input_pipeline import humaneval_infilling_to_testcases as he


def _write_jsonl(path: Path, tasks):
    with path.open("w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")


def test_convert_humaneval_to_tasks_direct_error(tmp_path: Path):
    task = {
        "task_id": "HumanEval/0",
        "entry_point": "foo",
        "prompt": "def foo():",
        "suffix": "    return 1",
        "canonical_solution": "    x = 1\n    return x",
        "test": "assert foo() == 1",
    }
    input_file = tmp_path / "he.jsonl"
    _write_jsonl(input_file, [task])

    args = he.Args(
        input_file=str(input_file),
        output_dir=str(tmp_path / "out"),
        base_path="/project/tasks",
        versions="direct,error",
        limit=-1,
    )

    tasks = he.convert_humaneval_to_tasks(args)
    assert len(tasks) == 2

    direct = next(tc for tc in tasks if tc["task_id"].startswith("humaneval_direct/"))
    error = next(tc for tc in tasks if tc["task_id"].startswith("humaneval_error/"))

    assert direct["states"][-1]["eval"] == "EVAL"
    assert error["states"][-1]["eval"] == "EVAL"

    expected_final = "def foo():\n    x = 1\n    return x\n    return 1"
    direct_files = direct["states"][-1]["files"]
    assert direct_files["/project/tasks/foo.py"] == expected_final

    assert direct["humaneval_meta"]["original_task_id"] == "HumanEval/0"
    assert error["humaneval_meta"]["test"] == "assert foo() == 1"


def test_create_continuation_task_short_solution_returns_none():
    parsed = {
        "task_id": "HumanEval/1",
        "entry_point": "bar",
        "prompt": "def bar():\n    ",
        "suffix": "\n    return 2",
        "canonical_solution": "x",
        "test": "",
        "prompt_lines": ["def bar():", "    "],
        "suffix_lines": ["    return 2"],
        "solution_lines": ["x"],
    }

    assert he.create_continuation_task(parsed, "/project/tasks", he.random.Random(0)) is None


def test_write_yaml_tasks_replaces_slashes(tmp_path: Path):
    tasks = [
        {
            "task_id": "humaneval_direct/HumanEval/2",
            "states": [],
        }
    ]

    he.write_yaml_tasks(str(tmp_path), tasks)
    assert (tmp_path / "humaneval_direct__HumanEval__2.yaml").exists()
