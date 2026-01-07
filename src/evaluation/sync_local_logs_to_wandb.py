#!/usr/bin/env python3
"""
Sync local evaluation logs to wandb.

This script reads local logs created by sglang_eval.py when using --use_local_logger
and uploads them to wandb as a single run with all data points.

Usage:
    python sync_local_logs_to_wandb.py --log_dir data/eval/local_logs/eval_<RUN_ID>
    
Or to sync all runs in a directory:
    python sync_local_logs_to_wandb.py --log_dir data/eval/local_logs --sync_all
"""

import argparse
import json
import os
import sys

import wandb


def load_metadata(log_dir: str) -> dict:
    """Load run metadata from the log directory."""
    metadata_file = os.path.join(log_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file, "r") as f:
        return json.load(f)


def load_metrics(log_dir: str) -> list:
    """Load all metrics from the JSONL file."""
    metrics_file = os.path.join(log_dir, "metrics.jsonl")
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    metrics = []
    with open(metrics_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))

    return metrics


def sync_single_run(log_dir: str, dry_run: bool = False) -> bool:
    """
    Sync a single run's local logs to wandb.

    Returns True if successful, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Syncing: {log_dir}")
    print(f"{'='*60}")

    try:
        # Load metadata and metrics
        metadata = load_metadata(log_dir)
        metrics = load_metrics(log_dir)

        if not metrics:
            print(f"  Warning: No metrics found in {log_dir}")
            return False

        # Sort metrics by eval_step
        metrics.sort(key=lambda x: x.get("eval_step", 0))

        print(f"  Run ID: {metadata['run_id']}")
        print(f"  Run Name: {metadata['run_name']}")
        print(f"  Project: {metadata['project']}")
        print(f"  Tags: {metadata.get('tags', [])}")
        print(f"  Data points: {len(metrics)}")
        print(f"  Steps: {[m.get('eval_step', 'N/A') for m in metrics]}")

        if dry_run:
            print("  [DRY RUN] Would upload to wandb")
            return True

        # Initialize wandb run
        wandb.init(
            project=metadata["project"],
            name=metadata["run_name"],
            id=metadata["run_id"],
            config=metadata.get("config", {}),
            tags=metadata.get("tags", []),
            resume="allow",  # Resume if exists, create if not
        )

        # Log each metric with its step
        for metric_entry in metrics:
            # Remove timestamp for wandb logging
            entry = {k: v for k, v in metric_entry.items() if k != "timestamp"}
            wandb.log(entry)

        wandb.finish()
        print(f"  ✓ Successfully synced to wandb!")
        return True

    except Exception as e:
        print(f"  ✗ Error syncing {log_dir}: {e}")
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
    # Sync a single run
    python sync_local_logs_to_wandb.py --log_dir data/eval/local_logs/eval_13032805
    
    # Sync all runs in a directory
    python sync_local_logs_to_wandb.py --log_dir data/eval/local_logs --sync_all
    
    # Dry run to see what would be synced
    python sync_local_logs_to_wandb.py --log_dir data/eval/local_logs --sync_all --dry_run
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

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        print(f"Error: Log directory not found: {args.log_dir}")
        sys.exit(1)

    if args.sync_all:
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
            sync_single_run(args.log_dir, dry_run=args.dry_run)
        else:
            print(f"Error: {args.log_dir} does not appear to be a valid run directory")
            print("  (missing metadata.json)")
            print("\nHint: Use --sync_all to sync all runs in a parent directory")
            sys.exit(1)


if __name__ == "__main__":
    main()
