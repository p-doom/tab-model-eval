#!/usr/bin/env python3
"""
Parse lm_eval output and log metrics to WandB or LocalLogger.

Supports any lm_eval task (ifeval, gsm8k, mmlu, hellaswag, etc.).
Task names are auto-detected from the results JSON.

Usage:
    # Direct WandB logging (Berlin) - for ifeval
    python log_lmeval_to_wandb.py \
        --results_dir /path/to/eval/output/ifeval/RUN_ID \
        --wandb_id RUN_ID \
        --wandb_name RUN_NAME \
        --wandb_project crowd-pilot-miles \
        --wandb_eval_type ifeval-v1

    # For gsm8k or other tasks
    python log_lmeval_to_wandb.py \
        --results_dir /path/to/eval/output/gsm8k/RUN_ID \
        --wandb_id RUN_ID \
        --wandb_name RUN_NAME \
        --wandb_project crowd-pilot-miles \
        --wandb_eval_type gsm8k-v1

    # Local logging (Juelich)
    python log_lmeval_to_wandb.py \
        --results_dir /path/to/eval/output \
        --wandb_id RUN_ID \
        --wandb_name RUN_NAME \
        --wandb_project crowd-pilot-miles \
        --wandb_eval_type lmeval-v1 \
        --use_local_logger \
        --local_log_dir /path/to/local_logs
"""

import argparse
import json
import os
import re
from glob import glob
from typing import Dict, List, Optional, Any

import wandb

from .eval_utils import LocalLogger


def find_results_json(step_dir: str) -> Optional[str]:
    """Find the results JSON file in a step directory.

    Handles both:
    - Exact: results.json
    - Timestamped: results_2026-01-17T03-44-15.933275.json
    """
    # lm_eval typically creates: step_dir/default/results_<timestamp>.json
    # or: step_dir/local-chat-completions/results.json
    # or directly: step_dir/results.json
    patterns = [
        os.path.join(step_dir, "results.json"),
        os.path.join(step_dir, "results_*.json"),
        os.path.join(step_dir, "*", "results.json"),
        os.path.join(step_dir, "*", "results_*.json"),
        os.path.join(step_dir, "**", "results.json"),
        os.path.join(step_dir, "**", "results_*.json"),
    ]

    for pattern in patterns:
        matches = glob(pattern, recursive=True)
        if matches:
            # If multiple matches (e.g., multiple timestamped files), return the latest
            if len(matches) > 1:
                matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]
    return None


def parse_lmeval_results(results_file: str) -> tuple[Dict[str, Any], List[str]]:
    """
    Parse lm_eval results JSON and extract relevant metrics.

    Returns:
        metrics: Dict of metric_name -> value
        task_names: List of task names found in results
    """
    with open(results_file, "r") as f:
        data = json.load(f)

    metrics = {}
    task_names = []

    # lm_eval results structure:
    # {
    #   "results": {
    #     "task_name": {
    #       "metric_name,filter": value,
    #       ...
    #     }
    #   },
    #   "configs": {...},
    #   ...
    # }

    results = data.get("results", {})

    for task_name, task_metrics in results.items():
        task_names.append(task_name)
        for metric_key, value in task_metrics.items():
            # Skip non-numeric values and stderr metrics
            if not isinstance(value, (int, float)):
                continue
            if "_stderr" in metric_key:
                continue

            # Clean up metric name (remove filter suffix like ",none")
            clean_key = metric_key.split(",")[0] if "," in metric_key else metric_key

            # Create prefixed metric name: task_name/metric_name
            full_key = f"{task_name}/{clean_key}"
            metrics[full_key] = value

    return metrics, task_names


def discover_step_dirs(results_dir: str) -> List[tuple]:
    """
    Discover step directories in results_dir.

    Returns list of (step_number, step_dir_path) sorted by step.
    """
    steps = []

    # Look for step_* directories
    for entry in os.listdir(results_dir):
        entry_path = os.path.join(results_dir, entry)
        if os.path.isdir(entry_path):
            # Match step_0, step_100, step_1000, etc.
            match = re.match(r"step_(\d+)", entry)
            if match:
                step_num = int(match.group(1))
                steps.append((step_num, entry_path))

    # Sort by step number
    steps.sort(key=lambda x: x[0])
    return steps


def main():
    parser = argparse.ArgumentParser(description="Log lm_eval results to WandB")

    # Input
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Directory containing lm_eval output (with step_* subdirs)",
    )

    # WandB settings
    parser.add_argument("--wandb_id", required=True, help="WandB run ID")
    parser.add_argument("--wandb_name", required=True, help="WandB run name")
    parser.add_argument("--wandb_project", default="crowd-pilot-miles", help="WandB project")
    parser.add_argument(
        "--wandb_eval_type",
        default="lmeval-v1",
        help="Eval type for namespacing (e.g., ifeval-v1, gsm8k-v1)",
    )
    parser.add_argument("--wandb_tags", nargs="*", default=[], help="WandB tags")

    # Local logger settings
    parser.add_argument(
        "--use_local_logger", action="store_true", help="Use local logger instead of WandB"
    )
    parser.add_argument(
        "--local_log_dir", default="data/eval/local_logs", help="Local log directory"
    )

    args = parser.parse_args()

    # Discover step directories
    step_dirs = discover_step_dirs(args.results_dir)

    if not step_dirs:
        print(f"No step directories found in {args.results_dir}")
        return

    print(f"Found {len(step_dirs)} step directories: {[s[0] for s in step_dirs]}")

    # Collect all tasks found across steps (for config)
    all_tasks = set()

    # Initialize logger (deferred for local logger until we know tasks)
    wandb_run = None
    logger = None

    if not args.use_local_logger:
        wandb_run = wandb.init(
            id=args.wandb_id,
            name=args.wandb_name,
            project=args.wandb_project,
            tags=args.wandb_tags,
            resume="allow",
        )

    # Process each step
    for step_num, step_dir in step_dirs:
        results_file = find_results_json(step_dir)

        if not results_file:
            print(f"Warning: No results.json found in {step_dir}, skipping step {step_num}")
            continue

        print(f"Processing step {step_num}: {results_file}")

        try:
            metrics, task_names = parse_lmeval_results(results_file)
            all_tasks.update(task_names)

            # Initialize local logger on first successful parse (now we know tasks)
            if args.use_local_logger and logger is None:
                logger = LocalLogger(
                    log_dir=args.local_log_dir,
                    run_id=args.wandb_id,
                    run_name=args.wandb_name,
                    project=args.wandb_project,
                    eval_type=args.wandb_eval_type,
                    config={
                        "eval_type": args.wandb_eval_type,
                        "tasks": list(all_tasks),
                        "results_dir": args.results_dir,
                    },
                    tags=args.wandb_tags,
                )

            # Add eval_step
            metrics["eval_step"] = step_num

            # Prefix metrics with eval type (task name already in key from parse)
            prefixed_metrics = {
                f"{args.wandb_eval_type}/{k}": v for k, v in metrics.items() if k != "eval_step"
            }
            prefixed_metrics["eval_step"] = step_num

            # Log
            if args.use_local_logger and logger:
                logger.log(prefixed_metrics)
            elif wandb_run:
                wandb.log(prefixed_metrics)

            print(f"  Logged {len(metrics)} metrics for step {step_num} (tasks: {task_names})")

        except Exception as e:
            print(f"Error processing step {step_num}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Finish
    if args.use_local_logger and logger:
        logger.finish()
    elif wandb_run:
        wandb.finish()

    print(f"Done! Logged {len(step_dirs)} steps. Tasks found: {sorted(all_tasks)}")


if __name__ == "__main__":
    main()
