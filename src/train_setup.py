from pathlib import Path
import json
import pandas as pd
import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.model import DiagnosticModel
from src.data import DicomDataset, VisualParams, BalancedBatchSampler
from src.brain_transforms import get_spatial_transform_list, get_color_transform_list

"""
Sets up a single training, given a dictionary of arguments, a list of person_ids, 
and a specific run_output_dir
"""

def setup(args_dict: dict, person_ids: list, run_output_dir: Path, data_samples_df: pd.DataFrame):
    """
    args_dict (dict) : dictionary of all relevant passed arguments
    people    (list) : [[train_people], [val_people], [test_people]]
    
    """
    print("Performing Setup")
    
    # Parse Parameters
    batch_size = args_dict['batch_size']
    num_workers = args_dict['num_workers']

    # Create Dataset for Train/Validation/Test & Apply Augmentations
    vis_params = VisualParams(display_method = args_dict['display_method'], 
                     norm_method = args_dict['norm_method'], 
                     masked_norm = args_dict['masked_norm'],
                     percentile_norm   = args_dict['perc_norm'])

    train_people = np.random.choice(person_ids['train'], size = int(len(person_ids['train']) * args_dict['trainset_frac'])).tolist()
    
    train_dataset = DicomDataset(data_samples_df, vis_params = vis_params, person_ids = train_people, summarize_name = 'train')
    val_dataset   = DicomDataset(data_samples_df, vis_params = vis_params, person_ids = person_ids['val'], summarize_name = 'val')
    test_dataset  = DicomDataset(data_samples_df, vis_params = vis_params, person_ids = person_ids['test'], summarize_name = 'test')

    augmentation_list = []
    if 's' in args_dict['aug']:
        print("Applying Spatital Augmentations")
        augmentation_list.extend(get_spatial_transform_list().copy())
    if 'c' in args_dict['aug']:
        print("Applying Color Augmentations")
        augmentation_list.extend(get_color_transform_list().copy())

    train_dataset.set_aug(augmentation_list)
    
    # Save JSON of run data distributions
    with open(run_output_dir / 'info.json', 'w') as f:
        json.dump( 
            {
                "train_counts": train_dataset.get_counts(),
                "val_counts": val_dataset.get_counts(),
                "test_counts": test_dataset.get_counts(),

            }, f, indent = 2
        )

    # Get Class Weights
    class_weights, per_sample_weights = train_dataset.get_weights()

    # 2a) Re-Sampling for Training
    shuffle = False
    if args_dict['balance'] == 'w':
        sampler = WeightedRandomSampler(weights=per_sample_weights, num_samples=len(per_sample_weights), replacement=True)
    elif args_dict['balance'] == 'b':
        sampler = BalancedBatchSampler(train_dataset.get_labels(), batch_size = batch_size)
    elif args_dict['balance'] == 'o':
        sampler = None
        shuffle = True
    elif args_dict['balance'] == '':
        sampler = None
        shuffle = True
    else: 
        raise ValueError(f"Unknown balance method: {args_dict['balance']}. Acceptable are one of w,b,o,''")
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = sampler, num_workers=num_workers, shuffle = shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler = None, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers = num_workers, shuffle=False)

    #4) Create Model
    model = DiagnosticModel(model_name = args_dict['model'], 
                            in_channels = args_dict['in_channels'], 
                            include_weights = args_dict['use_weights'])
    model = model.to(args_dict['device'])

    # 5) Create Loss Function
    if args_dict['balance'] == 'o':
        criterion = nn.CrossEntropyLoss(weight = class_weights.to(args_dict['device']))
    else:
        criterion = nn.CrossEntropyLoss()

    return model, (train_loader, val_loader, test_loader), criterion


