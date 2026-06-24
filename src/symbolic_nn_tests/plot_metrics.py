import os
import shutil

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

EXPERIMENT_TO_DATASET = {
    "experiment1": "QMNIST",
    "experiment2": "PubChem",
    "experiment3": "QMNIST",
    "experiment4": "QMNIST",
}


def get_test_metrics_by_dataset(base_dir="logs"):
    """Extracts test_acc and test_loss grouped by dataset."""
    results = {}
    if not os.path.exists(base_dir):
        return results

    for exp_dir in sorted(os.listdir(base_dir)):
        if exp_dir not in EXPERIMENT_TO_DATASET:
            continue

        dataset = EXPERIMENT_TO_DATASET[exp_dir]
        if dataset not in results:
            results[dataset] = {}

        exp_path = os.path.join(base_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue

        for run in sorted(os.listdir(exp_path)):
            run_dir = os.path.join(exp_path, run)
            if not os.path.isdir(run_dir):
                continue

            ea = event_accumulator.EventAccumulator(run_dir)
            ea.Reload()
            scalars = ea.Tags().get("scalars", [])

            metrics = {}
            if "test_acc" in scalars:
                metrics["test_acc"] = ea.Scalars("test_acc")[-1].value
            if "test_loss" in scalars:
                metrics["test_loss"] = ea.Scalars("test_loss")[-1].value

            if metrics:
                # Store the run with its experiment prefix to avoid collisions
                run_id = f"{exp_dir}/{run}"
                results[dataset][run_id] = metrics

    return results


def generate_and_log_summary_plots(base_dir="logs"):
    """Generates a bar chart of all test metrics per dataset and logs to Tensorboard."""
    dataset_metrics = get_test_metrics_by_dataset(base_dir)
    if not dataset_metrics:
        print(f"No test metrics found in {base_dir}.")
        return

    # Clear old summary to prevent appending
    summary_dir = os.path.join(base_dir, "summary")
    if os.path.exists(summary_dir):
        shutil.rmtree(summary_dir)

    for dataset, runs_data in dataset_metrics.items():
        if not runs_data:
            continue

        runs = list(runs_data.keys())
        accs = [runs_data[r].get("test_acc", 0.0) for r in runs]
        losses = [runs_data[r].get("test_loss", 0.0) for r in runs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Clean up run names for x-axis display
        display_runs = [
            r.replace("experiment1/", "e1: ")
            .replace("experiment2/", "e2: ")
            .replace("experiment3/", "e3: ")
            .replace("experiment4/", "e4: ")
            .replace("differentiable_", "")
            .replace("_constraints", "")
            .replace("_cross_entropy", "")
            for r in runs
        ]

        # Plot Accuracies
        ax1.bar(display_runs, accs, color="skyblue", edgecolor="black")
        ax1.set_title(f"{dataset} - Test Accuracy (or R2Score)", fontsize=14, pad=15)
        ax1.set_ylabel("Accuracy", fontsize=12)
        ax1.tick_params(axis="x", rotation=45, labelsize=8)
        ax1.set_ylim(0, max(accs) * 1.15 if accs and max(accs) > 0 else 1.0)
        for i, v in enumerate(accs):
            ax1.text(
                i,
                v + (max(accs) * 0.02 if accs else 0),
                f"{v:.4f}",
                ha="center",
                fontweight="bold",
                fontsize=8,
            )

        # Plot Losses
        ax2.bar(display_runs, losses, color="salmon", edgecolor="black")
        ax2.set_title(f"{dataset} - Test Loss", fontsize=14, pad=15)
        ax2.set_ylabel("Loss", fontsize=12)
        ax2.tick_params(axis="x", rotation=45, labelsize=8)
        ax2.set_ylim(0, max(losses) * 1.15 if losses and max(losses) > 0 else 1.0)
        for i, v in enumerate(losses):
            ax2.text(
                i,
                v + (max(losses) * 0.02 if losses else 0),
                f"{v:.4f}",
                ha="center",
                fontweight="bold",
                fontsize=8,
            )

        plt.tight_layout()

        dataset_summary_dir = os.path.join(summary_dir, dataset)
        os.makedirs(dataset_summary_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=dataset_summary_dir)
        writer.add_figure(f"Summary/{dataset}_Metrics", fig, global_step=0)
        writer.close()

        print(
            f"Summary bar chart for {dataset} generated and written "
            f"to {dataset_summary_dir} (Images tab)"
        )


if __name__ == "__main__":
    generate_and_log_summary_plots()
