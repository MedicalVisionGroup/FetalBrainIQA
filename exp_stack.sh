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
#SBATCH --array=0-11

# activate virtual environment
source /data/vision/polina/users/marcusbl/miniconda3/bin/activate bin_class
export PYTHONPATH="/data/vision/polina/users/marcusbl/bin_class:${PYTHONPATH}"

## EXECUTION OF PYTHON CODE:
cd /data/vision/polina/users/marcusbl/bin_class/src
cmds=(
  'python -m train --out_dir  mm_stack_untrain              --aug s --model resnet50 --epochs 250 --balance b --norm_method "min-max" --masked_norm --data_split_seed 1 --num_runs 6 --k_fold --batch_size 32 --mask_method "stack"'
  'python -m train --out_dir  mm_stack2_untrain             --aug s --model resnet50 --epochs 250 --balance b --norm_method "min-max" --masked_norm --data_split_seed 2 --num_runs 6 --k_fold --batch_size 32 --mask_method "stack"'
  'python -m train --out_dir  mm_stack        --use_weights --aug s --model resnet50 --epochs 200 --balance b --norm_method "min-max" --masked_norm --data_split_seed 1 --num_runs 6 --k_fold --batch_size 32 --mask_method "stack"'
  'python -m train --out_dir  mm_stack2       --use_weights --aug s --model resnet50 --epochs 200 --balance b --norm_method "min-max" --masked_norm --data_split_seed 2 --num_runs 6 --k_fold --batch_size 32 --mask_method "stack"'

  'python -m train --out_dir  mm_none_untrain               --aug s --model resnet50 --epochs 250 --balance b --norm_method "min-max" --masked_norm --data_split_seed 1 --num_runs 6 --k_fold --batch_size 32'
  'python -m train --out_dir  mm_none2_untrain              --aug s --model resnet50 --epochs 250 --balance b --norm_method "min-max" --masked_norm --data_split_seed 2 --num_runs 6 --k_fold --batch_size 32'
  'python -m train --out_dir  mm_none         --use_weights --aug s --model resnet50 --epochs 200 --balance b --norm_method "min-max" --masked_norm --data_split_seed 1 --num_runs 6 --k_fold --batch_size 32'
  'python -m train --out_dir  mm_none2        --use_weights --aug s --model resnet50 --epochs 200 --balance b --norm_method "min-max" --masked_norm --data_split_seed 2 --num_runs 6 --k_fold --batch_size 32'

  'python -m train --out_dir  mm_2stack_untrain              --aug s --model resnet50 --epochs 250 --balance b --norm_method "min-max" --masked_norm --data_split_seed 1 --num_runs 6 --k_fold --batch_size 32 --mask_method "stack2"'
  'python -m train --out_dir  mm_2stack2_untrain             --aug s --model resnet50 --epochs 250 --balance b --norm_method "min-max" --masked_norm --data_split_seed 2 --num_runs 6 --k_fold --batch_size 32 --mask_method "stack2"'
  'python -m train --out_dir  mm_2stack        --use_weights --aug s --model resnet50 --epochs 200 --balance b --norm_method "min-max" --masked_norm --data_split_seed 1 --num_runs 6 --k_fold --batch_size 32 --mask_method "stack2"'
  'python -m train --out_dir  mm_2stack2       --use_weights --aug s --model resnet50 --epochs 200 --balance b --norm_method "min-max" --masked_norm --data_split_seed 2 --num_runs 6 --k_fold --batch_size 32 --mask_method "stack2"'

)

eval ${cmds[$SLURM_ARRAY_TASK_ID]}


# python -m train --out_dir test1 --data_split_seed 42 --aug sc --use_tqdm --balance b --model resnet50 --epochs 100 --use_weights --norm_method min-max --masked_norm --perc_norm 0.02 --num_runs 5