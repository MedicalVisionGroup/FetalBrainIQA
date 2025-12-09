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
#SBATCH --array=0-0

# activate virtual environment
source /data/vision/polina/users/marcusbl/miniconda3/bin/activate bin_class
export PYTHONPATH="/data/vision/polina/users/marcusbl/bin_class:${PYTHONPATH}"

## EXECUTION OF PYTHON CODE:
cd /data/vision/polina/users/marcusbl/bin_class/src
cmds=(
  # 'python -m train --out_dir temp --aug s --model resnet50 --epochs 1 --balance b --use_weights --norm_method "min-max" --num_runs 5 --use_tqdm'
  # 'python -m train --out_dir batch_16  --aug s --model resnet50 --epochs 100 --balance b --use_weights --norm_method "min-max" --masked_norm --data_split_seed 1 --num_runs 6 --k_fold --batch_size 16'
  # 'python -m train --out_dir batch_32  --aug s --model resnet50 --epochs 100 --balance b --use_weights --norm_method "min-max" --masked_norm --data_split_seed 1 --num_runs 6 --k_fold --batch_size 32'
  # 'python -m train --out_dir batch_64  --aug s --model resnet50 --epochs 100 --balance b --use_weights --norm_method "min-max" --masked_norm --data_split_seed 1 --num_runs 6 --k_fold --batch_size 64'
  'python -m train --out_dir batch_128 --aug s --model resnet50 --epochs 100 --balance b --use_weights --norm_method "min-max" --masked_norm --data_split_seed 1 --num_runs 6 --k_fold --batch_size 128'
)

eval ${cmds[$SLURM_ARRAY_TASK_ID]}