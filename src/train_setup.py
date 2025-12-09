from pathlib import Path
import os
import json

import argparse
from dotenv import load_dotenv

import torch
import torch.nn as nn

from model import DiagnosticModel
from data import DicomDataset, BalancedBatchSampler, split_dataset
from augs_list import get_spatial_transform_list, get_color_transform_list
from torch.utils.data import DataLoader, WeightedRandomSampler

# ----- FIXED PARAMS -------
batch_size = 32
num_workers = 8
lr = 1e-4

num_train = 20
num_val = 5
num_test = 5
val_metric = 'f1'

def parse_args():
    print("Parsing Arguments")

    #1) Load .env for data/output directories
    load_dotenv("/data/vision/polina/users/marcusbl/bin_class/.env")
    data_dir = Path(os.environ["DATA_DIR"])
    output_root = Path(os.environ["OUTPUT_DIR_ROOT"])

    # 2) Parse the args
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",       # default if not given
        help="Subdirectory inside of outputs to save results"
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
        help= """
                1) w - resample w/ inv freq weights
                2) o - update the objective w/ the weights
                3) b - balance the training process exactly to 50-50
              """
    )
    parser.add_argument(
        "--model",
        type=str,
        default='resnet18',
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
        "--mask_method",
        type=str,
        help = """ 
                1) stack - make input to model scan + mask = 2 input channel
                2) mask - actually apply the mask [dataset limited to images w/ actual masks]
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
        "--balance_val",
        action="store_true",
        help = "Will balance validation set to be 50-50 at each sampling"
    )
    parser.add_argument(
        "--data_split_seed",
        type=int,
        default = -1,
        help = "The seed used for splitting the data"
    )
    

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict['batch_size'] = batch_size
    args_dict['num_workers'] = num_workers
    args_dict['lr'] = lr
    args_dict['dataset_cnts'] = [num_train, num_val, num_test]
    args_dict['val_metric'] = val_metric
    args_dict['in_channels'] =  2 if args.mask_method == 'stack' else 3
    args_dict['data_dir'] = data_dir
    args_dict['output_dir'] = output_root / args.out_dir

    return args_dict

def setup(args_dict: dict, people: list):
    """
    args_dict (dict) : dictionary of all relevant passed arguments
    people    (list) : [[train_people], [val_people], [test_people]]
    
    """
    print("Performing Setup")
    # Parse Parameters
    batch_size = args_dict['batch_size']
    num_workers = args_dict['num_workers']

    # Create Output Directory
    output_dir = args_dict['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")

    with open(output_dir / 'params.json', 'w') as f:
        args_copy = args_dict.copy()
        args_copy['output_dir'] = str(args_copy['output_dir'])
        args_copy['data_dir'] = str(args_copy['data_dir'])
        json.dump(args_copy, f, indent=2)

    # 2) Create Dataset for Train/Validation & Apply Augmentations
    dataset = DicomDataset(args_dict['data_dir'])
    dataset.set_norm(mask_method = args_dict['mask_method'], 
                     norm_method = args_dict['norm_method'], 
                     masked_norm = args_dict['masked_norm'],
                     perc_norm   = args_dict['perc_norm'])
    
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, people)
    augmentation_list = []
    if 's' in args_dict['aug']:
        print("Applying Spatital Augmentations")
        augmentation_list.extend(get_spatial_transform_list())
    if 'c' in args_dict['aug']:
        print("Applying Color Augmentations")
        augmentation_list.extend(get_color_transform_list(args_dict['mask_method']))

    train_dataset.set_aug(augmentation_list)
    
    # train_dataset.save_examples(output_dir, num_examples = 10)
    dataset.test_data_collect(output_dir = output_dir) # testing the scans / masks align

    dataset.summarize(name = "Original")
    train_dataset.summarize(name = "Train")
    val_dataset.summarize(name = "Val")
    test_dataset.summarize(name = "Test")

    # Get Class Weights
    class_weights, sample_weights = train_dataset.get_class_weights()

    # 2a) Re-Sampling for Training
    shuffle = False
    if args_dict['balance'] == 'w':
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    elif args_dict['balance'] == 'b':
        sampler = BalancedBatchSampler(train_dataset.get_labels(), batch_size = batch_size)
    elif args_dict['balance'] == 'o':
        sampler = None
        shuffle = True
    else: 
        raise ValueError(f"Unknown balance method: {args_dict['balance']}. Acceptable are one of w,b,o")
    
    val_sampler = None
    if args_dict['balance_val']:
        print('Will do validation balancing')
        val_sampler = BalancedBatchSampler(val_dataset.get_labels(), batch_size = batch_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = sampler, num_workers=num_workers, shuffle = shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler = val_sampler, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers = num_workers, shuffle=False)

    # 3) Device
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device("cuda")  # GPU -- should be used if using cluster

    print(f"Using device {device}")

    #4) Create Model
    model = DiagnosticModel(model_name = args_dict['model'], 
                            in_channels = args_dict['in_channels'], 
                            include_weights = args_dict['use_weights'])
    model = model.to(device)

    # 5) Create Loss Function
    if args_dict['balance'] == 'o':
        criterion = nn.CrossEntropyLoss(weight = class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    return model, (train_loader, val_loader, test_loader), criterion

