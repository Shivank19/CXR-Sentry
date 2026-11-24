import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_loss_curves(run_dir: Path) -> None:
    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("metrics.csv contains no completed epochs")

    epochs = [int(row["epoch"]) for row in rows]
    train_losses = [float(row["train_loss"]) for row in rows]
    dev_losses = [float(row["dev_loss"]) for row in rows]

    checkpoint_config = run_dir / "best_checkpoint" / "config.json"
    with checkpoint_config.open(encoding="utf-8") as file:
        best = json.load(file)

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(epochs, train_losses, "o-", label="Train")
    ax.plot(epochs, dev_losses, "s-", label="Dev")
    ax.axvline(
        best["epoch"],
        color="0.45",
        linestyle="--",
        label=f"Selected epoch ({best['epoch']})",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy per target token")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.savefig(run_dir / "loss_curves.png", dpi=300)
    fig.savefig(run_dir / "loss_curves.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a CXR-Sentry training run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    plot_loss_curves(args.run_dir)
    print(f"Saved loss_curves.png and loss_curves.svg in {args.run_dir}")