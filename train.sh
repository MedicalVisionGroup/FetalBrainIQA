#!/bin/bash
## SLURM Variables:
#SBATCH --job-name binary_classification
#SBATCH --output=/data/vision/polina/users/marcusbl/bin_class/slurm_outputs
#SBATCH -e /data/vision/polina/users/marcusbl/bin_class/slurm_outputs/%x-%j.err
#SBATCH -o /data/vision/polina/users/marcusbl/bin_class/slurm_outputs/%x-%j.out
#SBATCH --partition=polina-all
#SBATCH -A vision-polina
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --nodes=1
#SBATCH --qos=vision-polina-main
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-3

# activate virtual environment
source /data/vision/polina/users/marcusbl/miniconda3/bin/activate bin_class
export PYTHONPATH="/data/vision/polina/users/marcusbl/bin_class:${PYTHONPATH}"

## EXECUTION OF PYTHON CODE:
cmds=(
  "python -m src.train --out_dir resnet50_no_mask --aug sc --resample --model resnet50 --epochs 40"
  "python -m src.train --out_dir resnet50_mask --aug sc --resample --model resnet50 --epochs 40"
  "python -m src.train --out_dir resnet50_mask_channel_weight --aug sc --resample --model resnet50 --epochs 40 --inc_mask_channel --use_weights"
  "python -m src.train --out_dir resnet50_mask_channel_unweight --aug sc --resample --model resnet50 --epochs 40 --inc_mask_channel"
)

# python -m src.train --out_dir temp --aug sc --resample --model resnet18 --use_tqdm --epochs 3
eval ${cmds[$SLURM_ARRAY_TASK_ID]}






