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
#SBATCH --array=0-7

# activate virtual environment
source /data/vision/polina/users/marcusbl/miniconda3/bin/activate bin_class
export PYTHONPATH="/data/vision/polina/users/marcusbl/bin_class:${PYTHONPATH}"

## EXECUTION OF PYTHON CODE:
cd /data/vision/polina/users/marcusbl/bin_class/src
cmds=(
  'python -m train --out_dir test_mm02_1 --use_weights --data_split_seed 1 --aug sc --balance b --model resnet50 --epochs 50 --norm_method min-max     --masked_norm --perc_norm 0.02 --num_runs 5'
  'python -m train --out_dir test_mm02_2 --use_weights --data_split_seed 2 --aug sc --balance b --model resnet50 --epochs 50 --norm_method min-max     --masked_norm --perc_norm 0.02 --num_runs 5'
  'python -m train --out_dir test_mm00_1 --use_weights --data_split_seed 1 --aug sc --balance b --model resnet50 --epochs 50 --norm_method min-max     --masked_norm --num_runs 5'
  'python -m train --out_dir test_mm00_2 --use_weights --data_split_seed 2 --aug sc --balance b --model resnet50 --epochs 50 --norm_method min-max     --masked_norm --num_runs 5'
  'python -m train --out_dir test_ps02_1 --use_weights --data_split_seed 1 --aug sc --balance b --model resnet50 --epochs 50 --norm_method peak-squash --masked_norm --perc_norm 0.02 --num_runs 5'
  'python -m train --out_dir test_ps02_2 --use_weights --data_split_seed 2 --aug sc --balance b --model resnet50 --epochs 50 --norm_method peak-squash --masked_norm --perc_norm 0.02 --num_runs 5'
  'python -m train --out_dir test_ps00_1 --use_weights --data_split_seed 1 --aug sc --balance b --model resnet50 --epochs 50 --norm_method peak-squash --masked_norm --num_runs 5'
  'python -m train --out_dir test_ps00_2 --use_weights --data_split_seed 2 --aug sc --balance b --model resnet50 --epochs 50 --norm_method peak-squash --masked_norm --num_runs 5'
)
eval ${cmds[$SLURM_ARRAY_TASK_ID]}