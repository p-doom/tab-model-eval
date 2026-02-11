import yaml

from src.evaluation import eval_metrics


def test_compute_file_similarity_basic():
    assert eval_metrics.compute_file_similarity("a", "a") == 1.0
    assert eval_metrics.compute_file_similarity("", "a") == 0.0
    assert eval_metrics.compute_file_similarity("a", "") == 0.0


def test_compute_line_diff_metrics_additions_only():
    expected = "a\nb"
    predicted = "a\nb\nc"
    diff = eval_metrics.compute_line_diff_metrics(expected, predicted)
    assert diff["num_additions"] == 1
    assert diff["num_deletions"] == 0
    assert diff["total_diff_lines"] == 1


def test_evaluate_sample_with_prediction_error():
    result = eval_metrics.evaluate_sample(
        expected_files={"a.txt": "hello"},
        predicted_files=None,
        predicted_raw="",
        prediction_error="failed",
    )
    assert result["avg_similarity"] == 0.0
    assert result["files_evaluated"] == 0
    assert result["file_exact_match"] is False


def test_evaluate_sample_with_files():
    expected_files = {"a.txt": "hello\nworld"}
    predicted_files = {"a.txt": "hello\nworld!"}
    result = eval_metrics.evaluate_sample(
        expected_files=expected_files,
        predicted_files=predicted_files,
        predicted_raw="sed -i '2c\\world!' a.txt",
        prediction_error=None,
    )
    assert result["files_evaluated"] == 1
    assert result["num_unexpected_files"] == 0
    assert result["file_exact_match"] is False
    assert 0.0 < result["avg_similarity"] < 1.0
    assert len(result["file_details"]) == 1


def test_evaluate_sample_with_unexpected_file_counts():
    expected_files = {"a.txt": "hello"}
    predicted_files = {"a.txt": "hello", "b.txt": "extra"}
    result = eval_metrics.evaluate_sample(
        expected_files=expected_files,
        predicted_files=predicted_files,
        predicted_raw="ok",
        prediction_error=None,
    )
    assert result["num_unexpected_files"] == 1


def test_evaluate_task_aggregates_samples():
    expected_files = {"a.txt": "hello"}
    predicted_ok = {"a.txt": "hello"}
    predicted_bad = {"a.txt": "world"}

    task_result = {
        "task_id": "t1",
        "states": [{"eval": "EVAL", "files": expected_files}],
        "samples": [
            {"sample_idx": 0, "predicted_files": predicted_ok, "predicted_raw": "ok"},
            {"sample_idx": 1, "predicted_files": predicted_bad, "predicted_raw": "bad"},
        ],
    }

    result = eval_metrics.evaluate_task(task_result)
    assert result["num_samples"] == 2
    assert result["num_file_exact_match"] == 1
    assert result["file_exact_match_rate"] == 0.5
    assert result["pass_at_1"] == 1


def test_evaluate_task_no_samples():
    task_result = {"task_id": "t1", "states": [], "samples": []}
    result = eval_metrics.evaluate_task(task_result)
    assert result["num_samples"] == 0
    assert result["avg_similarity"] == 0.0
    assert result["pass_at_1"] == 0


def test_run_metrics_eval_writes_output(tmp_path):
    generations_file = tmp_path / "generations.yaml"
    metrics_file = tmp_path / "metrics.yaml"

    yaml_data = {
        "config": {"model": "test"},
        "results": [
            {
                "task_id": "t1",
                "states": [{"eval": "EVAL", "files": {"a.txt": "hello"}}],
                "samples": [
                    {
                        "sample_idx": 0,
                        "predicted_files": {"a.txt": "hello"},
                        "predicted_raw": "ok",
                    }
                ],
            }
        ],
    }

    generations_file.write_text(yaml.dump(yaml_data))

    scores = eval_metrics.run_metrics_eval(
        generations_file=str(generations_file),
        metrics_file=str(metrics_file),
        eval_step=0,
        limit=-1,
    )

    assert metrics_file.exists()
    assert scores["total_tasks"] == 1
    assert scores["total_samples"] == 1
