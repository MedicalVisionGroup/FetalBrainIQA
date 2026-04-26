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
cd /data/vision/polina/users/marcusbl/bin_class/src
cmds=(
  'python -m train --out_dir balance/b      --model resnet50 --use_weights --data_split_seed 1 --aug s --balance b --epochs 50 --norm_method min-max --display_method stack3 --masked_norm --num_runs 10'
  'python -m train --out_dir balance/o      --model resnet50 --use_weights --data_split_seed 1 --aug s --balance o --epochs 50 --norm_method min-max --display_method stack3 --masked_norm --num_runs 10'
  'python -m train --out_dir balance/w      --model resnet50 --use_weights --data_split_seed 1 --aug s --balance w --epochs 50 --norm_method min-max --display_method stack3 --masked_norm --num_runs 10'
  'python -m train --out_dir balance/none   --model resnet50 --use_weights --data_split_seed 1 --aug s --epochs 50 --norm_method min-max --display_method stack3 --masked_norm --num_runs 10'
)

eval ${cmds[$SLURM_ARRAY_TASK_ID]}


