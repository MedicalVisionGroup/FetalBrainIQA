from pathlib import Path
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import roc_curve
from torch.utils.data import Dataset

from src.data import split_people, get_samples_df
from src.train_setup import setup

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
    """
    Saves ROC.png and ROC.json files to the test_dir 
    """
    for raw_data_path in test_dir.iterdir():
        if raw_data_path.suffix != '.csv': 
            continue
        
        df = pd.read_csv(raw_data_path)
        fpr, tpr, thresholds = roc_curve(df['labels'], df['probs'])

        name = raw_data_path.stem[:-4]

        fig = plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC Curve on Test Data w/ {name}")
        plt.savefig(test_dir / f'{name}_ROC.png')
        plt.close(fig)

        # Save JSON of thresholds/fpr/tpr
        with open(test_dir / f'{name}_ROC.json', 'w') as f:
            json.dump({
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(), 
                'ts': thresholds.tolist()
            }, f, indent = 2)


def save_misclassifications(test_dir: Path, model_name: str, test_dataset: Dataset, use_tqdm: bool = True):
    """
    Saves two PDF's -- one for FP and one for FN -- into the test_dir
    """
    results_df = pd.read_csv(test_dir / f'{model_name}_raw.csv', index_col = 0)


    fp_df = results_df[(results_df['labels'] == 0) & (results_df['preds'] == 1)]
    fn_df = results_df[(results_df['labels'] == 1) & (results_df['preds'] == 0)]

    def save_pdf(df: pd.DataFrame, name: str):
        df = df.sort_values(by = 'probs')
        indices = df['idxs'].tolist()
        probs = df['probs'].tolist()

        with PdfPages(test_dir / f"{model_name}_{name}.pdf") as pdf:

            for start in tqdm(list(range(0, len(indices), 9)), f"Creating PDF for {model_name} {name}", disable = not use_tqdm):
                fig, axes = plt.subplots(3, 3, figsize = (15, 15))
                
                for i in range(3):
                    for j in range(3):
                        cnt = start + (i * 3) + j
                        if cnt >= len(indices):
                            axes[i,j].axis('off')
                            continue

                        idx = indices[cnt]
                        prob = probs[cnt]

                        img, mask, label, _ = test_dataset[idx]
                        if name == 'fp':
                            assert label == 0
                        else:                
                            assert label == 1        
                        
                        axes[i,j].imshow(img[0, :, :], cmap="gray", vmin=0, vmax=1)
                        axes[i,j].set_title(f"Index: {idx} | Prob: {prob}")

                        axes[i,j].axis("off")


                plt.tight_layout()

                pdf.savefig(fig)
                plt.close(fig)
    

    save_pdf(fp_df, "fp")
    save_pdf(fn_df, "fn")
    
    

if __name__ == '__main__':
    # Displaying Metrics & ROC
    output_dir = Path('/data/vision/polina/users/marcusbl/bin_class/outputs_experiment/test1')

    df = get_metric_for_all_runs(output_dir)

    display_metrics(output_dir, metrics = ['auc', 'tpr', 'fpr'], name = "epoch_metrics")
    display_metrics(output_dir, metrics = ['loss'], name = "loss_curve")

    display_roc(output_dir / 'run0' / 'test_info')

    # Testing the Save Bad Examples! (Have to do some annoying setup to get the Test Dataset...)
    run = 0
    model_name = 'model_loss'


    with open(output_dir / 'params.json') as f:
        args = json.load(f)

    data_samples_df, person_ids = get_samples_df(args['data_path'])
    people_groups = split_people(person_ids, fractions = args['split_fracs'], 
                                seed = args['data_split_seed'], num_runs = args['num_runs'])

    model, loaders, criterion = setup(args, people_groups[run], output_dir / f'run{run}', data_samples_df)
    _, _, test_loader = loaders

    save_misclassifications(output_dir / f'run{run}' / 'test_info', model_name, test_loader.dataset)