# Evaluating Image Quality for Fetal Brain MRI with Deep Learning

## 📁 Directory Structure

```bash
.
├── src/                    
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── brain_transforms.py
│   ├── train_setup.py
│   ├── evaluate.py
│   ├── display_utils.py
│   ├── train.py            # Main entry point
│   ├── parse_args.py
│   │
│   ├── results/            # Notebooks for analyzing experiment results 
│   └── label_sessions/     # Very unorganized code for Data Label Sessions  
│ 
├── train_scripts/          # Files (sh) for experiments 
├── thesis_figs/            # Generate figures for thesis
├── old_code/               # Unused and possibly buggy
│ 
├── environment.yaml        # Conda environment
├── .env                    # Example environment variables
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

## Setup

### Download the Repo
```
git clone git@github.com:MedicalVisionGroup/FetalBrainIQA.git
```

### Environment Variables
Create an empty `.env` file:
```bash
touch .env
```

1. Set `DATA_PATH` to the `/data/vision/polina/users/marcusbl/bin_class/label_sessions_data/label_session_3-11/final.csv`. This is a CSV file of all the data that we have (labled and unlabeled) that links to other data locations such as `/data/vision/polina/projects/fetal/common-data/BRAIN-IQA` and `/data/vision/polina/users/marcusbl/data/`. 
2. Set `OUTPUT_DIR_ROOT` to wherever you want to store your results.


### Conda Environment
Run:
```bash
conda env create -f environment.yml
conda activate bin_class
```

## Running the Code
Running the code is as simple as running `python -m src.train` with a variety of different parameters following. See the `.sh` files in `train_scripts` for examples. In addition, `src/parse_args.py` contains a detailed description of all of the parameters that the program accepts. 

Results will be stored in the `OUTPUT_DIR_ROOT` that you specified in the `.env` folder. In addition, one the parameters to the scripts requires an output directory name. The training will be run multiple times, and the results will be saved for each run. 

## Results

The results of each run will be saved in its corresponding output directory. To read these results clearly, go to `src/results/results_metrics.ipynb`, input the `output_dir`, `model_name`, and `metrics` of interest. Then run the rest of the notebook. This will display detailed results of all of the experiments in the given `output_dir`. It will also calculate statistical signficance. 

## Generating Thesis Figs

Run all of the notebooks in `thesis_figs`, inputting the correct directories if they are required, and all of the results will be saved into the `thesis_figs/figs` folder. This can be directly dragged into the LaTeX document for my thesis, and all of the figures will update accordingly. 

<!-- <!-- ## Set up the enviornment
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
``` -->

<!-- ### 2b) Submit a Job
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
 --> -->
