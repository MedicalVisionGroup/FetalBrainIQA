import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.metrics import roc_curve

def get_metric_for_all_runs(output_dir: Path) -> pd.DataFrame:
    all_dfs = []

    for run_dir in output_dir.iterdir():
        if not run_dir.is_dir():
            continue

        file_path = run_dir / "epoch_metrics" / "data.jsonl"
        with open(file_path, "r") as f:
            rows = [json.loads(line) for line in f]

        df = pd.DataFrame(rows)
        run_id = int(run_dir.stem.split("run")[1])

        train_df = pd.json_normalize(df["train"])
        val_df = pd.json_normalize(df["val"])

        # Add run as metric BEFORE concat
        train_df["run"] = run_id
        val_df["run"] = run_id

        df = pd.concat(
            {
                "train": train_df,
                "val": val_df,
            },
            axis=1,
        )

        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def display_metrics(output_dir: Path, metrics: list[str], name: str):
    df = get_metric_for_all_runs(output_dir)

    plt.figure(figsize=(8, 6))

    # Automatically generate distinct colors
    cmap = plt.get_cmap("tab10")
    colors = {metric: cmap(i) for i, metric in enumerate(metrics)}

    for metric in metrics:
        color = colors[metric]

        for split in ["train", "val"]:
            epochs = df[split]["epoch"]
            values = df[split][metric]
            runs = df[split]["run"]

            temp = pd.DataFrame({
                "epoch": epochs,
                "run": runs,
                "value": values
            })

            grouped = temp.groupby("epoch")["value"]

            mean = grouped.mean()
            std = grouped.std()
            n = grouped.count()

            margin = 1.96 * std / np.sqrt(n)

            linestyle = "-" if split == "train" else "--"

            plt.plot(
                mean.index,
                mean.values,
                label=f"{metric} ({split})",
                color=color,
                linestyle=linestyle,
                linewidth=2
            )

            plt.fill_between(
                mean.index,
                mean - margin,
                mean + margin,
                color=color,
                alpha=0.15
            )

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}.png", dpi=150)
    plt.close()

def display_roc(test_dir: Path):
    for raw_data_path in test_dir.iterdir():
        if raw_data_path.suffix != '.csv': 
            continue
        
        df = pd.read_csv(raw_data_path)
        fpr, tpr, _ = roc_curve(df['labels'], df['probs'])

        name = raw_data_path.stem[:-4]
        print(name)

        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC Curve on Test Data w/ {name}")
        plt.savefig(test_dir / f'{name}_ROC.png')



if __name__ == '__main__':
    output_dir = Path('/data/vision/polina/users/marcusbl/bin_class/outputs_experiment/test1')

    df = get_metric_for_all_runs(output_dir)

    print(df)
    print(df['train'])
    print(df['val'])

    display_metrics(output_dir, metrics = ['auc', 'tpr', 'fpr'], name = "epoch_metrics")
    display_metrics(output_dir, metrics = ['loss'], name = "loss_curve")

    display_roc(output_dir / 'run0' / 'test_info')

