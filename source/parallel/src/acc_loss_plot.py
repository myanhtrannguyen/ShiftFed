import pandas as pd
import matplotlib.pyplot as plt

FILE = "mpi_sync_lenet5_30r.csv"
# DATASET = ["MNIST", "SVHN", "USPS"]
DATASET = ["Global"]
METRIC = "acc"
# DATASET = ["SVHN"]
OUTPUT_FILE = "global_acc.png"
TITLE = "Global Accuracy"

RANK_FILTERS = {
    "svhn": (1, 3),
    "usps": (4, 7),
    "mnist": (8, 11)
}


def plot_metric(
    csv_file,
    datasets,
    metric="acc",
    output_file=None,
    title=None,
    figsize=(8, 5)
):

    df = pd.read_csv(csv_file)

    plt.figure(figsize=figsize)

    for dataset in datasets:

        col = f"{dataset.lower()}_{metric}"

        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")

        if dataset.lower() == "global":

            metric_df = (
                df.groupby("round")[col]
                .mean()
                .reset_index()
            )

        else:

            rank_min, rank_max = RANK_FILTERS[dataset.lower()]

            metric_df = (
                df[
                    (df["rank"] >= rank_min)
                    & (df["rank"] <= rank_max)
                ]
                .groupby("round")[col]
                .mean()
                .reset_index()
            )

        plt.plot(
            metric_df["round"],
            metric_df[col],
            linewidth=2,
            label=dataset.upper()
        )

    plt.xlabel("Round")

    if metric == "acc":
        plt.ylabel("Accuracy (%)")
    else:
        plt.ylabel("Loss")

    if title is None:
        title = "Accuracy" if metric == "acc" else "Loss"

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()

plot_metric(
    FILE,
    DATASET,
    metric=METRIC,
    output_file=OUTPUT_FILE,
    title=TITLE,
    figsize=(8, 5)
)