import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


CONDITIONS = ("matched", "shuffled", "constant")

def main():
    parser = argparse.ArgumentParser(
        description="Summarize and plot an image-dependence evaluation."
    )
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    summary = []
    for condition in CONDITIONS:
        group = [row for row in rows if row["condition"] == condition]
        tokens = [int(row["n_reference_tokens"]) for row in group]
        if not group or sum(tokens) == 0:
            raise ValueError(f"No scored reference tokens for {condition}")

        # Each row stores a per-study mean. Weight it by the study's token count to recover the mean across all reference tokens.
        total_nll = sum( float(row["reference_mean_nll"]) * count for row, count in zip(group, tokens))

        changed = sum(row["draft_changed_vs_matched"].strip().lower() == "true" for row in group)

        summary.append({
            "condition": condition,
            "n_studies": len(group),
            "n_reference_tokens": sum(tokens),
            "reference_nll_per_token": total_nll / sum(tokens),
            "drafts_changed_vs_matched": changed if condition != "matched" else None,
        })

    prefix = args.input.stem
    out_dir = args.input.parent
    summary_path = out_dir / f"{prefix}_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    fig, (loss_ax, change_ax) = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")

    losses = [row["reference_nll_per_token"] for row in summary]
    loss_bars = loss_ax.bar(
        ["Matched", "Shuffled", "Constant"],
        losses,
        color=["#196A9C", "#8E9CA6", "#8E9CA6"],
    )
    loss_ax.bar_label(loss_bars, fmt="%.3f", padding=3)
    loss_ax.set_ylim(0, max(losses) * 1.15)
    loss_ax.set_ylabel("Reference NLL per token")

    # Matched is the comparison point, so show only the altered conditions.
    altered = summary[1:]
    change_rates = [
        100 * row["drafts_changed_vs_matched"] / row["n_studies"]
        for row in altered
    ]
    change_bars = change_ax.bar(
        ["Shuffled", "Constant"], change_rates, color="#8E9CA6"
    )
    change_ax.bar_label(
        change_bars,
        labels=[
            f"{row['drafts_changed_vs_matched']}/{row['n_studies']}"
            for row in altered
        ],
        padding=3,
    )
    change_ax.set_ylim(0, 115)
    change_ax.set_yticks([0, 25, 50, 75, 100])
    change_ax.set_ylabel("Drafts changed vs. matched (%)")

    for extension in ("png", "svg"):
        fig.savefig(out_dir / f"{prefix}.{extension}", dpi=300)
    plt.close(fig)
    print(f"Saved {summary_path} and PNG/SVG figures")


if __name__ == "__main__":
    main()
