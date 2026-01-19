#!/usr/bin/env python3
"""
Sync local evaluation logs to wandb.

This script reads local logs created by the eval scripts when using --use_local_logger
and uploads them to wandb.

Supports two directory structures:

1. Legacy (single eval):
   {log_dir}/{run_id}/
       metadata.json
       metrics.jsonl

2. Multi-eval (parallel execution):
   {log_dir}/{run_id}/
       metadata.json
       unit_tests/metrics.jsonl
       handcrafted/metrics.jsonl
       humaneval/metrics.jsonl
       ifeval/metrics.jsonl

Usage:
    # Sync a single run (aggregates all evals into one wandb run)
    python sync_local_logs_to_wandb.py --log_dir data/eval/local_logs/13032805
    
    # Sync all runs in a directory
    python sync_local_logs_to_wandb.py --log_dir data/eval/local_logs --sync_all
    
    # Dry run to see what would be synced
    python sync_local_logs_to_wandb.py --log_dir data/eval/local_logs --sync_all --dry_run
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import wandb


# Known eval types (subdirectories to look for)
EVAL_TYPES = [
    "unit_tests",
    "handcrafted",
    "humaneval",
    "ifeval",
    "unit-test-v1",
    "handcrafted-v1",
    "humaneval-direct-v1",
    "ifeval-v1",
]


def load_metadata(run_dir: str) -> dict:
    """Load run metadata from the run directory."""
    metadata_file = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file, "r") as f:
        return json.load(f)


def load_metrics_from_file(metrics_file: str) -> List[dict]:
    """Load metrics from a single JSONL file."""
    if not os.path.exists(metrics_file):
        return []

    metrics = []
    with open(metrics_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))
    return metrics


def find_eval_subdirs(run_dir: str) -> List[Tuple[str, str]]:
    """
    Find all eval-type subdirectories in a run directory.

    Returns list of (eval_type, subdir_path) tuples.
    """
    eval_subdirs = []

    for item in os.listdir(run_dir):
        item_path = os.path.join(run_dir, item)
        if os.path.isdir(item_path):
            # Check if this subdir has a metrics.jsonl file
            metrics_file = os.path.join(item_path, "metrics.jsonl")
            if os.path.exists(metrics_file):
                eval_subdirs.append((item, item_path))

    return sorted(eval_subdirs)


def load_all_metrics(run_dir: str) -> Tuple[List[dict], Dict[str, int]]:
    """
    Load all metrics from a run directory.

    Supports both legacy (single metrics.jsonl) and multi-eval (subdirectories) structures.

    Returns:
        (all_metrics, eval_counts) where eval_counts maps eval_type -> number of data points
    """
    all_metrics = []
    eval_counts = {}

    # Check for legacy structure (metrics.jsonl in run_dir)
    legacy_metrics_file = os.path.join(run_dir, "metrics.jsonl")
    if os.path.exists(legacy_metrics_file):
        metrics = load_metrics_from_file(legacy_metrics_file)
        if metrics:
            all_metrics.extend(metrics)
            eval_counts["legacy"] = len(metrics)

    # Check for multi-eval structure (subdirectories with metrics.jsonl)
    eval_subdirs = find_eval_subdirs(run_dir)
    for eval_type, subdir_path in eval_subdirs:
        metrics_file = os.path.join(subdir_path, "metrics.jsonl")
        metrics = load_metrics_from_file(metrics_file)
        if metrics:
            all_metrics.extend(metrics)
            eval_counts[eval_type] = len(metrics)

    return all_metrics, eval_counts


def sync_single_run(
    run_dir: str,
    dry_run: bool = False,
    wandb_id_override: Optional[str] = None,
    wandb_name_override: Optional[str] = None,
) -> bool:
    """
    Sync a single run's local logs to wandb.

    Aggregates all eval subdirectories into a single wandb run.

    Args:
        run_dir: Path to the run directory
        dry_run: If True, don't actually upload
        wandb_id_override: Override the wandb run ID (useful for appending to training run)
        wandb_name_override: Override the wandb run name

    Returns True if successful, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Syncing: {run_dir}")
    print(f"{'='*60}")

    try:
        # Load metadata
        metadata = load_metadata(run_dir)

        # Load all metrics (from legacy and/or subdirectories)
        all_metrics, eval_counts = load_all_metrics(run_dir)

        if not all_metrics:
            print(f"  Warning: No metrics found in {run_dir}")
            return False

        # Sort metrics by eval_step
        all_metrics.sort(key=lambda x: x.get("eval_step", 0))

        # Use overrides if provided
        wandb_id = wandb_id_override or metadata["run_id"]
        wandb_name = wandb_name_override or metadata["run_name"]

        print(f"  Run ID: {metadata['run_id']}")
        print(f"  Run Name: {metadata['run_name']}")
        print(f"  WandB ID: {wandb_id}")
        print(f"  WandB Name: {wandb_name}")
        print(f"  Project: {metadata['project']}")
        print(f"  Tags: {metadata.get('tags', [])}")
        print(f"  Total data points: {len(all_metrics)}")
        print(f"  Eval types found:")
        for eval_type, count in eval_counts.items():
            print(f"    - {eval_type}: {count} data points")
        print(f"  Steps: {sorted(set(m.get('eval_step', 'N/A') for m in all_metrics))}")

        if dry_run:
            print("  [DRY RUN] Would upload to wandb")
            return True

        # Initialize wandb run
        wandb.init(
            project=metadata["project"],
            name=wandb_name,
            id=wandb_id,
            config=metadata.get("config", {}),
            tags=metadata.get("tags", []),
            resume="allow",  # Resume if exists, create if not
        )

        # Log each metric with its step
        for metric_entry in all_metrics:
            # Remove timestamp for wandb logging
            entry = {k: v for k, v in metric_entry.items() if k != "timestamp"}
            wandb.log(entry)

        wandb.finish()
        print(f"  Successfully synced to wandb!")
        return True

    except Exception as e:
        print(f"  Error syncing {run_dir}: {e}")
        import traceback

        traceback.print_exc()
        return False


def find_run_dirs(base_dir: str) -> list:
    """Find all run directories in the base directory."""
    run_dirs = []

    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if it's a valid run directory (has metadata.json)
            metadata_file = os.path.join(item_path, "metadata.json")
            if os.path.exists(metadata_file):
                run_dirs.append(item_path)

    return sorted(run_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Sync local evaluation logs to wandb",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Sync a single run (aggregates all evals)
    python sync_local_logs_to_wandb.py --log_dir data/eval/local_logs/13032805
    
    # Sync all runs in a directory
    python sync_local_logs_to_wandb.py --log_dir data/eval/local_logs --sync_all
    
    # Dry run to see what would be synced
    python sync_local_logs_to_wandb.py --log_dir data/eval/local_logs --sync_all --dry_run
    
    # Override wandb ID to append to a specific run
    python sync_local_logs_to_wandb.py --log_dir data/eval/local_logs/13032805 --wandb_id my_training_run_id
""",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Path to the log directory (single run) or parent directory (with --sync_all)",
    )
    parser.add_argument(
        "--sync_all", action="store_true", help="Sync all runs found in the log_dir"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be synced without actually uploading",
    )
    parser.add_argument(
        "--wandb_id",
        type=str,
        default=None,
        help="Override wandb run ID (useful for appending to an existing training run)",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Override wandb run name",
    )

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        print(f"Error: Log directory not found: {args.log_dir}")
        sys.exit(1)

    if args.sync_all:
        if args.wandb_id:
            print("Warning: --wandb_id is ignored when using --sync_all")

        # Sync all runs in the directory
        run_dirs = find_run_dirs(args.log_dir)

        if not run_dirs:
            print(f"No valid run directories found in {args.log_dir}")
            sys.exit(1)

        print(f"Found {len(run_dirs)} run(s) to sync:")
        for d in run_dirs:
            print(f"  - {os.path.basename(d)}")

        success_count = 0
        for run_dir in run_dirs:
            if sync_single_run(run_dir, dry_run=args.dry_run):
                success_count += 1

        print(f"\n{'='*60}")
        print(f"Sync complete: {success_count}/{len(run_dirs)} runs synced successfully")
        print(f"{'='*60}")

    else:
        # Sync single run
        # Check if log_dir is the run directory or contains metadata.json
        if os.path.exists(os.path.join(args.log_dir, "metadata.json")):
            sync_single_run(
                args.log_dir,
                dry_run=args.dry_run,
                wandb_id_override=args.wandb_id,
                wandb_name_override=args.wandb_name,
            )
        else:
            print(f"Error: {args.log_dir} does not appear to be a valid run directory")
            print("  (missing metadata.json)")
            print("\nHint: Use --sync_all to sync all runs in a parent directory")
            sys.exit(1)


if __name__ == "__main__":
    main()
