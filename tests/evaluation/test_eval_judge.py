import asyncio
from types import SimpleNamespace

from src.evaluation import eval_judge


def test_format_files_for_prompt_with_content():
    formatted = eval_judge.format_files_for_prompt({"a.txt": "hi"})
    assert "### a.txt" in formatted
    assert "```" in formatted
    assert "hi" in formatted


def test_evaluate_task_skips_exact_and_empty():
    args = eval_judge.Args(skip_exact_matches=True, skip_empty_predictions=True)
    task_result = {
        "task_id": "t1",
        "states": [{"eval": "EVAL", "files": {"a.txt": "ok"}}],
        "samples": [
            {"sample_idx": 0, "exact_match": 1, "predicted_raw": "x"},
            {"sample_idx": 1, "exact_match": 0, "predicted_raw": "   "},
        ],
    }

    result = asyncio.run(
        eval_judge.evaluate_task(
            None,
            asyncio.Semaphore(1),
            task_result,
            system_prompt="system",
            prompt_template="{expected_files}\n{predicted_files}",
            args=args,
        )
    )
    assert result["num_samples"] == 2
    assert result["num_equivalent"] == 1
    assert result["num_skipped"] == 2
    assert result["num_judged"] == 0


class _FakeChatCompletions:
    async def create(self, **kwargs):
        message = SimpleNamespace(
            content='{"equivalent": 1, "reasoning": "ok"}', reasoning_content=""
        )
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class _FakeChat:
    def __init__(self):
        self.completions = _FakeChatCompletions()


class _FakeAsyncOpenAI:
    def __init__(self):
        self.chat = _FakeChat()


class _BadJsonChatCompletions:
    async def create(self, **kwargs):
        message = SimpleNamespace(content="not-json", reasoning_content="")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class _BadJsonChat:
    def __init__(self):
        self.completions = _BadJsonChatCompletions()


class _BadJsonAsyncOpenAI:
    def __init__(self):
        self.chat = _BadJsonChat()


def test_judge_single_sample_success():
    args = eval_judge.Args(max_attempts=1, num_judge_samples=1)
    result = asyncio.run(
        eval_judge.judge_single_sample(
            client=_FakeAsyncOpenAI(),
            sem=asyncio.Semaphore(1),
            task_id="t1",
            sample_idx=0,
            expected_files={"a.txt": "ok"},
            predicted_files={"a.txt": "ok"},
            assertions="",
            context="",
            system_prompt="system",
            prompt_template="{expected_files}\n{predicted_files}",
            args=args,
        )
    )

    assert result["equivalent"] == 1
    assert result["skipped"] is False


def test_judge_single_sample_no_prediction():
    args = eval_judge.Args(max_attempts=1, num_judge_samples=1)
    result = asyncio.run(
        eval_judge.judge_single_sample(
            client=_FakeAsyncOpenAI(),
            sem=asyncio.Semaphore(1),
            task_id="t1",
            sample_idx=0,
            expected_files={"a.txt": "ok"},
            predicted_files=None,
            assertions="",
            context="",
            system_prompt="system",
            prompt_template="{expected_files}\n{predicted_files}",
            args=args,
        )
    )
    assert result["skipped"] is True
    assert result["skip_reason"] == "no_prediction"


def test_judge_single_sample_invalid_json():
    args = eval_judge.Args(max_attempts=1, num_judge_samples=1)
    result = asyncio.run(
        eval_judge.judge_single_sample(
            client=_BadJsonAsyncOpenAI(),
            sem=asyncio.Semaphore(1),
            task_id="t1",
            sample_idx=0,
            expected_files={"a.txt": "ok"},
            predicted_files={"a.txt": "ok"},
            assertions="",
            context="",
            system_prompt="system",
            prompt_template="{expected_files}\n{predicted_files}",
            args=args,
        )
    )
    assert result["equivalent"] == 0
    assert "error" in result
