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

You can run code via `python train.py`. 

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

