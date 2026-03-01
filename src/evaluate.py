"""Evaluation script to aggregate metrics and generate comparison figures."""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import wandb


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Results directory"
    )
    parser.add_argument(
        "--run_ids", type=str, required=True, help="JSON list of run IDs"
    )
    return parser.parse_args()


def fetch_run_data(entity: str, project: str, run_id: str) -> dict:
    """
    Fetch run data from WandB API.

    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run display name

    Returns:
        Dictionary with config, summary, and history
    """
    api = wandb.Api()

    # Query runs by display name
    runs = api.runs(
        f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
    )

    if not runs:
        raise ValueError(f"No run found with display name: {run_id}")

    run = runs[0]  # Most recent run with that name

    # Get summary metrics
    summary = dict(run.summary)

    # Get config
    config = dict(run.config)

    # Get history (time series data)
    history = run.history()

    return {
        "run_id": run_id,
        "config": config,
        "summary": summary,
        "history": history,
        "url": run.url,
    }


def export_per_run_metrics(run_data: dict, output_dir: Path):
    """
    Export per-run metrics to JSON and generate figures.

    Args:
        run_data: Run data from WandB
        output_dir: Output directory for this run
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export metrics
    metrics = {
        "run_id": run_data["run_id"],
        "summary": run_data["summary"],
        "config": run_data["config"],
        "url": run_data["url"],
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Exported metrics: {output_dir / 'metrics.json'}")

    # Generate per-run figures
    history = run_data["history"]

    if not history.empty:
        # Plot samples over time
        if "num_samples" in history.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(history.index, history["num_samples"], marker="o", markersize=3)
            ax.set_xlabel("Problem Index")
            ax.set_ylabel("Num Samples")
            ax.set_title(f"{run_data['run_id']}: Samples per Problem")
            ax.grid(True, alpha=0.3)

            output_path = output_dir / "samples_per_problem.pdf"
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Generated figure: {output_path}")

        # Plot tokens over time
        if "tokens" in history.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(history.index, history["tokens"], marker="o", markersize=3)
            ax.set_xlabel("Problem Index")
            ax.set_ylabel("Tokens")
            ax.set_title(f"{run_data['run_id']}: Tokens per Problem")
            ax.grid(True, alpha=0.3)

            output_path = output_dir / "tokens_per_problem.pdf"
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Generated figure: {output_path}")


def generate_comparison_figures(all_run_data: list[dict], output_dir: Path):
    """
    Generate comparison figures across all runs.

    Args:
        all_run_data: List of run data dictionaries
        output_dir: Output directory for comparison figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract common metrics across runs
    metrics_to_compare = [
        "accuracy",
        "avg_samples_used",
        "avg_tokens_per_problem",
        "accuracy_per_1k_tokens",
    ]

    for metric in metrics_to_compare:
        fig, ax = plt.subplots(figsize=(10, 6))

        for run_data in all_run_data:
            run_id = run_data["run_id"]
            history = run_data["history"]

            # Map summary metric to history column
            metric_col = {
                "accuracy": "correct",
                "avg_samples_used": "num_samples",
                "avg_tokens_per_problem": "tokens",
            }.get(metric)

            if metric_col and metric_col in history.columns:
                # Plot time series
                ax.plot(
                    history.index,
                    history[metric_col],
                    label=run_id,
                    marker="o",
                    markersize=2,
                    alpha=0.7,
                )

        ax.set_xlabel("Problem Index")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Comparison: {metric.replace('_', ' ').title()}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_path = output_dir / f"comparison_{metric}.pdf"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Generated comparison figure: {output_path}")

    # Bar chart for summary metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_compare):
        ax = axes[idx]

        run_ids = []
        values = []

        for run_data in all_run_data:
            if metric in run_data["summary"]:
                run_ids.append(run_data["run_id"])
                values.append(run_data["summary"][metric])

        if values:
            colors = ["#2ecc71" if "proposed" in rid else "#3498db" for rid in run_ids]
            ax.bar(range(len(run_ids)), values, color=colors)
            ax.set_xticks(range(len(run_ids)))
            ax.set_xticklabels(run_ids, rotation=45, ha="right")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(metric.replace("_", " ").title())
            ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "comparison_summary.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated comparison summary: {output_path}")


def export_aggregated_metrics(all_run_data: list[dict], output_dir: Path):
    """
    Export aggregated comparison metrics.

    Args:
        all_run_data: List of run data dictionaries
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics by run_id
    metrics = {}
    for run_data in all_run_data:
        run_id = run_data["run_id"]
        metrics[run_id] = run_data["summary"]

    # Determine primary metric and best runs
    primary_metric = "accuracy"

    proposed_runs = {k: v for k, v in metrics.items() if "proposed" in k}
    baseline_runs = {k: v for k, v in metrics.items() if "comparative" in k}

    best_proposed = None
    best_baseline = None

    if proposed_runs:
        best_proposed = max(
            proposed_runs.items(), key=lambda x: x[1].get(primary_metric, 0)
        )

    if baseline_runs:
        best_baseline = max(
            baseline_runs.items(), key=lambda x: x[1].get(primary_metric, 0)
        )

    gap = None
    if best_proposed and best_baseline:
        gap = best_proposed[1].get(primary_metric, 0) - best_baseline[1].get(
            primary_metric, 0
        )

    aggregated = {
        "primary_metric": primary_metric,
        "metrics": metrics,
        "best_proposed": best_proposed[0] if best_proposed else None,
        "best_baseline": best_baseline[0] if best_baseline else None,
        "gap": gap,
    }

    with open(output_dir / "aggregated_metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"Exported aggregated metrics: {output_dir / 'aggregated_metrics.json'}")


def main():
    """Main evaluation function."""
    args = parse_args()

    results_dir = Path(args.results_dir)
    run_ids = json.loads(args.run_ids)

    # Get WandB config from environment or default
    entity = os.getenv("WANDB_ENTITY", "airas")
    project = os.getenv("WANDB_PROJECT", "ui-test-20260302")

    print(f"Fetching data for {len(run_ids)} runs from {entity}/{project}...")

    # Fetch all run data
    all_run_data = []
    for run_id in run_ids:
        try:
            run_data = fetch_run_data(entity, project, run_id)
            all_run_data.append(run_data)
            print(f"  Fetched: {run_id}")
        except Exception as e:
            print(f"  Warning: Failed to fetch {run_id}: {e}")

    if not all_run_data:
        print("Error: No run data found")
        return

    # Export per-run metrics and figures
    for run_data in all_run_data:
        run_output_dir = results_dir / run_data["run_id"]
        export_per_run_metrics(run_data, run_output_dir)

    # Generate comparison figures
    comparison_dir = results_dir / "comparison"
    generate_comparison_figures(all_run_data, comparison_dir)

    # Export aggregated metrics
    export_aggregated_metrics(all_run_data, comparison_dir)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
