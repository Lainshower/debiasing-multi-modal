"""
CelebA Dataset
- Reference code: https://github.com/kohpangwei/group_DRO/blob/master/data/celebA_dataset.py
- See Group DRO, https://arxiv.org/abs/1911.08731 for more
"""

import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CelebaEmbeddings(Dataset):
    def __init__(self, data_dir='./data/celeba/', split='train',
                 embedding_dir='./data/embeddings/celeba/RN50/embedding_prediction.json', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.embedding_dir = embedding_dir
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}

        self.metadata_df = pd.read_csv(os.path.join(self.data_dir, 'list_attr_celeba.csv'))
        self.split_df = pd.read_csv(os.path.join(self.data_dir, 'list_eval_partition.csv'))
        self.metadata_df['partition'] = self.split_df['partition'].copy().tolist()
        self.metadata_df = self.metadata_df[self.split_df['partition'] == self.split_dict[self.split]]


        self.embeddings_df = pd.read_json(self.embedding_dir) # key : image_filename
        indices_to_convert = ['blond', 'male', 'group', 'y_pred'] # str -> index
        self.embeddings_df.loc[indices_to_convert] = self.embeddings_df.loc[indices_to_convert].astype('int64')
        
        # Get the y values
        self.y_array = self.metadata_df['Blond_Hair'].values # Target Class (=Hair)
        self.confounder_array = self.metadata_df['Male'].values # Spurious Bias (=Gender) 
        self.y_array[self.y_array == -1] = 0
        self.confounder_array[self.confounder_array == -1] = 0
        self.group_array = (self.y_array * 2 + self.confounder_array).astype('int') # Group Label

        # Extract filenames and splits
        self.filename_array = self.metadata_df['image_id'].values
        self.split_array = self.metadata_df['partition'].values

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
        img_filename = self.filename_array[idx]
        # img_filename = os.path.join(self.data_dir, 'img_align_celeba/img_align_celeba', self.filename_array[idx])
        # img = Image.open(img_filename).convert('RGB')
        ebd_full = self.embeddings_df[img_filename]

        ebd_y = ebd_full['blond']
        ebd_y_group = ebd_full['group']
        ebd_y_spurious = ebd_full['male']
        ebd_y_pred = ebd_full['y_pred']
        ebd = torch.from_numpy(np.array(ebd_full['image_embedding'])).float()

        y = self.targets[idx]
        y_group = self.targets_group[idx]
        y_spurious = self.targets_spurious[idx]
        # y_split = self.split_array[idx]

        assert ((y==ebd_y) and (y_group==ebd_y_group) and (y_spurious==ebd_y_spurious)), f"inconsistency between {os.path.join(self.data_dir, 'metadata.csv')} and {self.embedding_dir}\n \
            Should be same: y: {y}=={ebd_y} | group: {y_group}=={ebd_y_group} | spurious_attribute: {y_spurious}=={ebd_y_spurious} "
        
        return ebd, {"y": y,"group": y_group,"spurious": y_spurious, "ebd_y_pred": ebd_y_pred}, img_filename  # img, (target class(hair), groups(4), Spurious Bias(gender) Split), idx

def get_transform_celeba():
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform


def load_celeba_embeddings(data_dir, embedding_dir, bs_train=512, bs_val=512, num_workers=8, transform=None):
    """
    Default dataloader setup for CelebA

    Args:
    - args (argparse): Experiment arguments
    - train_shuffle (bool): Whether to shuffle training data
    Returns:
    - (train_loader, val_loader, test_loader): Tuple of dataloaders for each split
    """
    train_set = CelebaEmbeddings(data_dir, 'train', embedding_dir, transform)
    train_loader = DataLoader(train_set, batch_size=bs_train, shuffle=True, num_workers=num_workers)

    val_set = CelebaEmbeddings(data_dir, 'train', embedding_dir, transform)
    val_loader = DataLoader(val_set, batch_size=bs_val, shuffle=False, num_workers=num_workers)

    test_set = CelebaEmbeddings(data_dir, 'train', embedding_dir, transform)
    test_loader = DataLoader(test_set, batch_size=bs_val, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader