"""
Embedding Loader
- for Linear Proving (Evaluation of featuue quality)
- for off-the-shelf modeul (such as Lin. Adapter or Contra. Adapter)
- ETC
"""

# function for checking whether splited validation sets have same group distribution
# from collections import Counter
# def count_unique_group_values(dataset):
#     group_counts = Counter()
#     for _, meta_data, _ in dataset:
#         group = meta_data['group'].item()
#         group_counts[group] += 1
#     return group_counts

import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class WaterbirdsEmbeddings(Dataset):
    def __init__(self, data_dir='./data/waterbirds/waterbird_complete95_forest2water2', split='train',
                 embedding_dir='./data/embeddings_unnormalized/waterbirds/RN50/embedding_prediction.json', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.embedding_dir = embedding_dir
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}

        self.metadata_df = pd.read_csv(os.path.join(self.data_dir, 'metadata.csv'))
        self.metadata_df = self.metadata_df[self.metadata_df['split'] == self.split_dict[self.split]]

        print(self.embedding_dir)
        self.embeddings_df = pd.read_json(self.embedding_dir) # key : image_filename
        indices_to_convert = ['y', 'place', 'group', 'y_pred', 'split'] # str -> int
        self.embeddings_df.loc[indices_to_convert] = self.embeddings_df.loc[indices_to_convert].astype('int64')
        
        # Get the y values
        self.y_array = self.metadata_df['y'].values  # Target Class (=Species)
        self.confounder_array = self.metadata_df['place'].values # Spurious Bias (=Place) 
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
        
        # For calcultating weighted test mean acc. (using training distribution.)
        self.group_counts = (torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()
        self.group_ratio = self.group_counts / len(self)
        
        # When indexing in contrastive batch sampling (중복되는 샘플 생길 경우 트리거)
        self.on_contrastive_batch = False



    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        img_filename = self.filename_array[idx]
        
        if not self.on_contrastive_batch: # on Normal Embedding Batch
            ebd_full = self.embeddings_df[img_filename]            
        else: # on Contrastive Batch
            ebd_full = self.embeddings_df.iloc[:, idx]
        
        ebd_y = ebd_full.loc['y']
        ebd_y_group = ebd_full.loc['group']
        ebd_y_spurious = ebd_full.loc['place']
        ebd_y_pred = ebd_full.loc['y_pred']
        # print(idx, img_filename)
        ebd = torch.from_numpy(np.array(ebd_full.loc['image_embedding'])).float()
        
        y = self.targets[idx]
        y_group = self.targets_group[idx]
        y_spurious = self.targets_spurious[idx]

        assert ((y==ebd_y) and (y_group==ebd_y_group) and (y_spurious==ebd_y_spurious)), f"inconsistency between {os.path.join(self.data_dir, 'metadata.csv')} and {self.embedding_dir}\n \
            Should be same: y: {y}=={ebd_y} | group: {y_group}=={ebd_y_group} | spurious_attribute: {y_spurious}=={ebd_y_spurious} "

        return ebd, {"class": y,"group": y_group,"spurious": y_spurious, "ebd_y_pred": ebd_y_pred}, img_filename
    
def stratified_split_dataset(dataset, test_size=0.5):
    """
    Splits a dataset in a stratified fashion using its targets, returning two Subset objects representing the train and test splits.
    """
    reg_idx, val_idx = train_test_split(
        np.arange(len(dataset.group_array)),
        test_size=test_size,
        random_state=42, # For Debugging 
        stratify=dataset.group_array)
    
    reg_subset = torch.utils.data.Subset(dataset, reg_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    return reg_subset, val_subset

def load_waterbirds_embeddings(data_dir, embedding_dir, bs_train=512, bs_val=256, num_workers=8, transform=None):
    train_set = WaterbirdsEmbeddings(data_dir, 'train', embedding_dir, transform)
    train_loader = DataLoader(train_set, batch_size=bs_train, shuffle=True, num_workers=num_workers)

    # val_set = WaterbirdsEmbeddings(data_dir, 'val', embedding_dir, transform)
    # val_loader = DataLoader(val_set, batch_size=bs_val, shuffle=False, num_workers=num_workers)
    
    val_set = WaterbirdsEmbeddings(data_dir, 'val', embedding_dir, transform)
    reg_vet, val_set = stratified_split_dataset(val_set, test_size=0.5)

    reg_loader = DataLoader(reg_vet, batch_size=bs_val, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=bs_val, shuffle=False, num_workers=num_workers)

    test_set = WaterbirdsEmbeddings(data_dir, 'test', embedding_dir, transform)
    test_loader = DataLoader(test_set, batch_size=bs_val, shuffle=False, num_workers=num_workers)

    return train_loader, reg_loader, val_loader, test_loader
