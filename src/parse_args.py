from dotenv import load_dotenv
import argparse
from pathlib import Path
import os
import shutil
import torch
import json

# ----- FIXED PARAMS -------
num_workers = 8
lr = 1e-4

frac_train = 0.70
frac_val = 0.15
frac_test = 0.15

def parse_args():
    print("Parsing Arguments")

    #1) Load .env for data/output directories
    load_dotenv("/data/vision/polina/users/marcusbl/bin_class/.env")
    data_path = Path(os.environ["DATA_PATH"])
    output_root = Path(os.environ["OUTPUT_DIR_ROOT"])
    output_root.mkdir(exist_ok=True)

    # 2) Parse the args
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",       # default if not given
        help="Subdirectory inside of output folder to save results"
    )
    parser.add_argument(
        "--aug",
        type=str,
        default='',
        help='Augmentation types: s-spatial, c-color'
    )
    parser.add_argument(
        "--use_tqdm",
        action="store_true",    # becomes True if flag is present
        help="Enable tqdm progress bars"
    )
    parser.add_argument(
        "--balance", 
        type=str,
        default = '',
        help= """
                1) w - resample w/ inv freq weights
                2) o - update the objective w/ the weights
                3) b - balance the batch exactly to 50-50
              """
    )
    parser.add_argument(
        "--model",
        type=str,
        default='resnet50',
        help='Specify the type of model using for classification'
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help='Specify the # of epochs using in training'
    )
    parser.add_argument(
        "--use_weights",
        action="store_true",
        help = "If true, downloads the weights for the model you're using from PyTorch"
    )
    parser.add_argument(
        "--display_method",
        type=str,
        default = "",
        help = """ 
                1) stack2 - make input to model: scan + mask = 2 input channel
                2) mask - actually apply the mask [dataset limited to images w/ actual masks]
                3) stack3 - make input to model: scan x 2 + mask = 3 input channel
                4) default is repeat the slice 3 times w/o a mask
               """,
    )
    parser.add_argument(
        "--norm_method",
        type=str,
        help = "Type of normalization method: min-max or peak-squash"
    )
    parser.add_argument(
        "--masked_norm",
        action="store_true",
        help = "True/False should normalize wrt the mask and not the entire image"
    )
    parser.add_argument(
        "--perc_norm",
        type=float,
        default=0,
        help = "The quantile for min-max normalization"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help = "The # of runs to run the training experimentation "
    )
    parser.add_argument(
        "--data_split_seed",
        type=int,
        default = -1,
        help = "The seed used for splitting the data"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32, 
        help = "The batch size"
    )
    parser.add_argument(
        "--k_fold",
        action="store_true",
        help = "Will use k_fold cross validation instead of random data on each round"
    )
    parser.add_argument(
        "--include_edges",
        action = "store_true",
        help = "If used, will use the edge slices and label them all as good"
    )
    parser.add_argument(
        "--label",
        type=str,
        default='final',
        help = """
                Specify which label to use for the experiment: 
                1. R = ramya label (dataset1)
                2. session1 = data from data session 1
                3. final = final data after data session 2
                """
    )
    parser.add_argument(
        "--trainset_frac", 
        type=float,
        default=1.0,
        help = "Fraction of available training dataset to use. Default is all, but can experiment w/ smaller"
    )

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict['dataset_types'] = ['BCH', 'R']
    args_dict['num_workers'] = num_workers
    args_dict['lr'] = lr
    args_dict['split_fracs'] = [frac_train, frac_val, frac_test]
    args_dict['in_channels'] =  2 if args.display_method == 'stack2' else 3
    args_dict['data_path'] = data_path
    args_dict['output_dir'] = Path(output_root / args.out_dir)


    # Clearning Output Directory
    if args_dict['output_dir'].exists():
        shutil.rmtree(args_dict['output_dir'])
    print(f"Results will be saved in: {args_dict['output_dir']}")
    args_dict['output_dir'].mkdir(exist_ok=True, parents = True)

    # Getting the Device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'  # GPU -- should be used if using cluster
    args_dict['device'] = device
    print(f"Using device {device}")

    # Dump a copy of args / params
    with open(args_dict['output_dir'] / 'params.json', 'w') as f:
        args_copy = args_dict.copy()
        args_copy['output_dir'] = str(args_copy['output_dir'])
        args_copy['data_path'] = str(args_copy['data_path'])
        json.dump(args_copy, f, indent=2)

    return args_dict
