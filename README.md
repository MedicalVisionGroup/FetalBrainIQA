# fetal-mri-quality-assessment
Building a ConvNet to assess quality of fetal MRI scans


# Setup & Login

## Important Directories:
- Data: `/data/vision/polina/projects/fetal/common-data/BRAIN-IQA`
- User: `/data/vision/polina/users/marcusbl/bin_class`

## 1) ssh 
- Using VSCode, choose remote ssh, `tig-slurm-direct`, and then type password/duo factor
- When remote, choose `tig-slurm`, and you'll have to do it twice. 
- note: these shortcuts are saved in `~/.ssh/config`

## 2) Running the Code
Screens allow you to run the code even when u disconnect from the cluster. Best practice is to create a separate shell for running the code. So open a new terminal. 

Create a screen:
```screen -S name_of_screen```

Resume a screen:
```screen -r name_of_screen```

Exit Screen:
```CTRL A+D```

### 2a) Interractive Shell

```
srun -p polina-all -A vision-polina --qos=vision-polina-main --gres=gpu:rtx_5000:1 -c 10 --mem=40G --time=0-10:00:00 --ntasks=1 --pty bash -l
```

Things should turn green, and you may need to activate conda:

```
conda init
source ~/.bashrc
conda activate bin_class
```

or 

```
miniconda3/bin/conda init
source ~/.bashrc
conda activate bin_class
cd bin_class
```

You can run code via 

```
cd src
python train.py
```

### 2b) Submit a Job
- See the `.sh` file in the repo
- We also clear the slurm_outputs folder with:
```
rm -f /data/vision/polina/users/marcusbl/bin_class/slurm_outputs/*.out
rm -f /data/vision/polina/users/marcusbl/bin_class/slurm_outputs/*.err
```
Submit with:
```
sbatch path_to_file.sh
```

You can show jobs with `squeue --user=marcusbl` and kill a job with `scancel job_id`

### 2c) SSH Mount
sshfs marcusbl@tig-slurm.csail.mit.edu://data/vision/polina/users/mfirenze/Data_sharing_MIT_Margherita /Users/marcusbluestone/Desktop/mnt1

# Understanding Results
Positive = Bad Scan; anomoly detected - something is wrong w/ the scan of the brain
Negative = Good Scan; nothing is detected

TPR = out of all positives, how many did the model CORRECTLY predict as positive? 
- High TPR (woohoo): you catch most of the positives
- We need a high TPR; otherwise, we will miss bad scans

FPR = out of all negatives, how many did the model INCORRECTLY predict as positive?
- High FPR (uh oh): you are overpredicting positives
- We need a low FPR; otherwise, we will force rescans when they aren't necessary

# Best Paramater Choices:
1. batch_size: 32. We experimented w/ all of them and they all gave basically the same results
2. aug: ?? (s or c or sc)
3. balance: b. We experimented w/ all options, and b / o gave best options, best b is best & simplest.
4. model: resenet50. Higher capacity and fast. 
5. epochs: 150. We want to see convergence, so we train for a long time, even after overfitting. 
6. use_weights: True! We want to use the weights of the ResNet and not train it from scratch!! (Testing if this is true for mask_method="stack" as well)
7. norm_method and masked_norm & perc_norm: ??
8. mask_method: ?? 
9. k_fold: use it for accurate results