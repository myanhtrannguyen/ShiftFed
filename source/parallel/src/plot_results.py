"""Plot FL accuracy/loss, timing breakdowns, and speedup."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FL MPI experiment results.")
    parser.add_argument("--log-dir", default="./outputs/fl_mpi")
    parser.add_argument("--out-dir", default="./outputs/fl_mpi/figures")
    return parser.parse_args()


def save_line_plot(df: pd.DataFrame, x: str, ys: list[str], title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for y in ys:
        if y in df.columns:
            ax.plot(df[x], df[y], marker="o", linewidth=1.5, label=y)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(path for path in log_dir.glob("*.csv") if path.name != "benchmark_summary.csv")
    if not csv_files:
        raise SystemExit(f"No experiment CSV files found in {log_dir}")

    frames = []
    for path in csv_files:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    round_df = (
        df.groupby(["source_file", "mode", "model", "round"], as_index=False)
        .agg(
            global_acc=("global_acc", "mean"),
            global_loss=("global_loss", "mean"),
            compute_time=("compute_time", "sum"),
            communication_time=("communication_time", "sum"),
        )
        .sort_values(["source_file", "round"])
    )

    for source_file, group in round_df.groupby("source_file"):
        stem = Path(source_file).stem
        save_line_plot(
            group,
            "round",
            ["global_acc", "global_loss"],
            f"Accuracy/Loss per round - {stem}",
            out_dir / f"{stem}_metrics.png",
        )
        save_line_plot(
            group,
            "round",
            ["compute_time", "communication_time"],
            f"Compute vs communication - {stem}",
            out_dir / f"{stem}_timing.png",
        )

    summary_path = log_dir / "benchmark_summary.csv"
    if summary_path.exists():
        import matplotlib.pyplot as plt

        summary = pd.read_csv(summary_path)
        labels = summary.apply(lambda row: f"{row['model']}-{int(row['rounds'])}r", axis=1)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, summary["speedup"])
        ax.set_title("MPI FL speedup vs sequential FL")
        ax.set_ylabel("speedup")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "speedup.png", dpi=160)
        plt.close(fig)

    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()
