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
#SBATCH --array=0-2

# activate virtual environment
source /data/vision/polina/users/marcusbl/miniconda3/bin/activate bin_class
export PYTHONPATH="/data/vision/polina/users/marcusbl/bin_class:${PYTHONPATH}"

## EXECUTION OF PYTHON CODE:
cd /data/vision/polina/users/marcusbl/bin_class/src
cmds=(
  # 'python -m train --out_dir temp --aug s --model resnet50 --epochs 1 --balance b --use_weights --norm_method "min-max" --num_runs 6 --use_tqdm'
  'python -m train --out_dir mm_stack2 --aug s --model resnet50 --epochs 150 --balance b --use_weights --norm_method "min-max" --masked_norm --data_split_seed 2 --num_runs 6 --k_fold --batch_size 32 --mask_method "stack"'
  'python -m train --out_dir mm_none2  --aug s --model resnet50 --epochs 150 --balance b --use_weights --norm_method "min-max" --masked_norm --data_split_seed 2 --num_runs 6 --k_fold --batch_size 32'
)

eval ${cmds[$SLURM_ARRAY_TASK_ID]}