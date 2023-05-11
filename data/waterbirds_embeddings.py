"""
Embedding Loader
- for Linear Proving (Evaluation of featuue quality)
- for off-the-shelf modeul (such as Lin. Adapter or Contra. Adapter)
- ETC
"""



import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class Embeddings(Dataset):
    def __init__(self, data_dir='./data/waterbirds/waterbird_complete95_forest2water2', split='train',
                 embedding_dir='./data/embeddings/waterbirds/RN50/embedding_prediction.json', transform=None, zs_group_label=None):
        self.data_dir = data_dir
        self.split = split
        self.embedding_dir = embedding_dir
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}

        self.metadata_df = pd.read_csv(os.path.join(self.data_dir, 'metadata.csv'))
        self.metadata_df = self.metadata_df[self.metadata_df['split'] == self.split_dict[self.split]]

        self.embeddings_df = pd.read_json(self.embedding_dir) # key : image_filename
        indices_to_convert = ['y', 'place', 'group', 'y_pred'] # str -> index
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

        # Attribute for noisy label detection
        self.noise_or_not = np.abs(self.y_array - self.confounder_array)  # 1 if minor (noisy)

        self.zs_group_label = zs_group_label
        if zs_group_label:
            self.preds_group_zeroshot = torch.tensor(np.load(zs_group_label))
            self.noise_or_not = (self.targets_group != self.preds_group_zeroshot).long().numpy()  # 1 if noisy

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        img_filename = self.filename_array[idx]
        # img_filename = os.path.join(self.data_dir, img_filename)
        # img = Image.open(img_filename).convert('RGB')
        ebd_full = self.embeddings_df[img_filename]

        ebd_y = ebd_full['y']
        ebd_y_group = ebd_full['group']
        ebd_y_spurious = ebd_full['place']
        ebd_y_pred = ebd_full['y_pred']
        ebd = torch.from_numpy(np.array(ebd_full['image_embedding'])).float()
        
        y = self.targets[idx]
        y_group = self.targets_group[idx]
        y_spurious = self.targets_spurious[idx]

        assert ((y==ebd_y) and (y_group==ebd_y_group) and (y_spurious==ebd_y_spurious)), f"inconsistency between {os.path.join(self.data_dir, 'metadata.csv')} and {self.embedding_dir}\n \
            Should be same: y: {y}=={ebd_y} | group: {y_group}=={ebd_y_group} | spurious_attribute: {y_spurious}=={ebd_y_spurious} "
        # if self.zs_group_label:
        #     y_group_zs = self.preds_group_zeroshot[idx]
        #     return x, (y, y_group, y_spur gious, y_group_zs), idx

        return ebd, {"y": y,"group": y_group,"spurious": y_spurious, "ebd_y_pred": ebd_y_pred}, img_filename
    


    
def load_embeddings(data_dir, embedding_dir, bs_train=512, bs_val=512, num_workers=8, transform=None, zs_group_label=None):
    train_set = Embeddings(data_dir, 'train', embedding_dir, transform, zs_group_label)
    train_loader = DataLoader(train_set, batch_size=bs_train, shuffle=True, num_workers=num_workers)

    val_set = Embeddings(data_dir, 'val', embedding_dir, transform, zs_group_label)
    val_loader = DataLoader(val_set, batch_size=bs_val, shuffle=False, num_workers=num_workers)

    test_set = Embeddings(data_dir, 'test', embedding_dir, transform, zs_group_label)
    test_loader = DataLoader(test_set, batch_size=bs_val, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
