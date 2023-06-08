"""
Waterbirds Dataset
- Reference code: https://github.com/kohpangwei/group_DRO/blob/master/data/cub_dataset.py
- See Group DRO, https://arxiv.org/abs/1911.08731 for more details



NOTE 수정사항 // waterbirds.py -> waterbirds_js.py (새벽이라 따로 물어보지 않고 잠시 고쳐서 사용함)
- Del: image_dir = 'combined' 
- Del: self.image_dir = image_dir
- Modify: data_dir = ['./data/waterbirds'] -> ['./data/waterbirds/waterbird_complete95_forest2water2'] (Official)

"""

import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class Waterbirds(Dataset):

    def __init__(self, data_dir='./data/waterbirds/waterbird_complete95_forest2water2', split='train', transform=None): # NOTE: modified
        self.data_dir = data_dir
        self.split = split
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}

        self.metadata_df = pd.read_csv(os.path.join(self.data_dir, 'metadata.csv')) 
        
        self.metadata_df = self.metadata_df[self.metadata_df['split'] == self.split_dict[self.split]]

        # Get the y values
        self.y_array = self.metadata_df['y'].values  # Target Class (=Species)
        self.confounder_array = self.metadata_df['place'].values # # Spurious Bias (=Place) 
        self.group_array = (self.y_array * 2 + self.confounder_array).astype('int') # Group Label

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
    
        self.targets = torch.tensor(self.y_array)
        self.targets_group = torch.tensor(self.group_array)
        self.targets_spurious = torch.tensor(self.confounder_array)

        self.transform = transform

        self.n_classes = 2
        self.n_groups = 4
        self.n_places = 2
        
        # NOTE for calcultating weighted test mean acc. (using training distribution.)
        self.group_counts = (torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()
        self.group_ratio = self.group_counts / len(self)
        
    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        # img_filename = os.path.join(self.data_dir, self.image_dir, self.filename_array[idx]) # NOTE: modified
        img_filename = os.path.join(self.data_dir,  self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        x = self.transform(img)

        y = self.targets[idx]
        y_group = self.targets_group[idx]
        y_spurious = self.targets_spurious[idx]
        y_split = self.split_array[idx]

        # if self.zs_group_label:
        #     y_group_zs = self.preds_group_zeroshot[idx]
        #     return x, (y, y_group, y_spurious, y_group_zs), idx

        return x, (y, y_group, y_spurious, y_split), img_filename # img, (target class(species), groups(4), Spurious Bias(place)), idx


def get_transform_cub(train):

    if not train:
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                (224, 224),
                scale=(0.7, 1.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def load_waterbirds(root_dir, bs_train=128, bs_val=128, num_workers=16):
    """
    Default dataloader setup for Waterbirds

    Args:
    - args (argparse): Experiment arguments
    - train_shuffle (bool): Whether to shuffle training data
    Returns:
    - (train_loader, val_loader, test_loader): Tuple of dataloaders for each split
    """
    train_set = Waterbirds(root_dir, split='train')
    train_loader = DataLoader(train_set, batch_size=bs_train, shuffle=True, num_workers=num_workers)

    val_set = Waterbirds(root_dir, split='val')
    val_loader = DataLoader(val_set, batch_size=bs_val, shuffle=False, num_workers=num_workers)

    test_set = Waterbirds(root_dir, split='test')
    test_loader = DataLoader(test_set, batch_size=bs_val, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
