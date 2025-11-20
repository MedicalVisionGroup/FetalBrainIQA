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
#SBATCH --array=0-5

# activate virtual environment
source /data/vision/polina/users/marcusbl/miniconda3/bin/activate bin_class
export PYTHONPATH="/data/vision/polina/users/marcusbl/bin_class:${PYTHONPATH}"

## EXECUTION OF PYTHON CODE:
cd /data/vision/polina/users/marcusbl/bin_class/src
cmds=(
  "python -m train --out_dir balance_o2 --aug s --model resnet50 --epochs 60 --balance o"
  "python -m train --out_dir balance_w2--aug s --model resnet50 --epochs 60 --balance w"
  "python -m train --out_dir balance_b2 --aug s --model resnet50 --epochs 60 --balance b"
  "python -m train --out_dir balance_o_download2 --aug s --model resnet50 --epochs 60 --balance o --use_weights"
  "python -m train --out_dir balance_w_download2 --aug s --model resnet50 --epochs 60 --balance w --use_weights"
  "python -m train --out_dir balance_b_download2 --aug s --model resnet50 --epochs 60 --balance b --use_weights"
)

eval ${cmds[$SLURM_ARRAY_TASK_ID]}