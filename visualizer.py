
from visualizer_supcon import *
import sys
from tqdm import tqdm
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm

def get_top_k_indices(array, k):
    indices = np.argsort(array)[::-1][:k]
    return indices

def find_closest_sample(dataset, anchor, top_k = 1):
    anchor_normalized = anchor / norm(anchor)
    dataset_normalized = dataset / norm(dataset, axis=1, keepdims=True)

    similarity_scores = np.dot(dataset_normalized, anchor_normalized)
    # 가장 가까운 샘플 인덱스 찾기
    closest_indices = get_top_k_indices(similarity_scores, k=top_k)
    
    return closest_indices

# import umap 
import umap.umap_ as umap # should be installed by "pip install umap-learn" (아마 내 기억엔 ㅋ..)

from torchvision.utils import make_grid
from sklearn.manifold import MDS
from torchvision import models, transforms
import torchvision
import json
import os
from copy import deepcopy
from easydict import EasyDict # pip install easydict

# from wb_data import WaterBirdsDataset, get_loader, get_transform_cub
class VisHandler():
    """
    - 1) Embedding 기반 Loader 받아서 Zero-shot prediction, Embeddings reduction 등 수행.
      - 1.1) Adapter 학습 하고 난 다음의 pth도 저장해놨으니, 불러와서 임베딩 똑같이 뽑을 수 있음.
      - 1.2) Linear probing / Adapter 등 학습 시 results/...에 결과 저장됨. 
    """
    def __init__(self, args):
        """
        Initialized by arguments for "single run"
        """
        self.args = args
        self.device = (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    
        # self.train_results = EasyDict(read_pickle_file(os.path.join(self.run_path, 'full_dict.pickle')))
        self.final_results = {} # Best Train / Val / Text
        
        if self.args.dataset == "waterbirds" :
            self.legend_labels_dict = {"target": {0: "Landbird", 1: "Waterbird"}, "spurious": {0: "Land-background", 1:"Water-background"}, 
                            "group": {0: "Landbird on Land-background", 1: "Landbird on Water-background",
                                        2: "Waterbird on Land-background", 3: "Waterbird on Water-background"},
                            "prediction": {0: "Pred. to Landbird",
                                            1: "Pred. to Waterbird"}}

        self.model = None
        self.text_embeddings = []
        self.group_wise_stat_ebd = {}
        self.group_wise_stat_conf = {}
        
        self.epoch = 0
    
    def SaveWaterbirdsDatasets(self, trainset):
        self.train_set = trainset
    
    def SaveWaterbirdsLoaders(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def SaveTextEmbeddings(self, embedding_dir):
        self.text_embeddings.extend(get_text_embedding(embedding_dir, return_key=True))
    
    def SaveModel(self, classifier):
        self.classifier = classifier
        
    def SaveUtils(self, criterion, get_yp_func, train_group_ratio):
        self.criterion = criterion
        self.get_yp_func = get_yp_func
        self.train_group_ratio = train_group_ratio
        
    def SaveZeroShotResults(self, train_loader, val_loader, test_loader):
        self.zs_results = {}
        _, _, train_group_acc = validate_zs(self.args, train_loader, self.classifier, self.criterion, self.get_yp_func, self.train_group_ratio, target="class", print_label='Get ZS Acc. of train (class)')    
        _, _, val_group_acc = validate_zs(self.args, val_loader, self.classifier, self.criterion, self.get_yp_func, self.train_group_ratio, target="class", print_label='Get ZS Acc. of val (class)')    
        _, _, test_group_acc = validate_zs(self.args, test_loader, self.classifier, self.criterion, self.get_yp_func, self.train_group_ratio, target="class", print_label='Get ZS Acc. of test (class)')    
        self.zs_results['train'] = train_group_acc
        self.zs_results['val'] = val_group_acc
        self.zs_results['test'] = test_group_acc
    
    def GetEmbeddings(self, dataloader):
        # # NOTE Adapter 학습 이후 모델 받아서 추출하는 라인 추가해야함.
        
        total_embeddings = []

        total_labels = []
        total_spuriouss = []
        total_groups = []
        # total_confidences = []
        total_predictions = [] # Zero-shot

        print('> Saving activations')

        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader, desc='Running inference')):
                embeddings, labels_dict, _= data
                labels = labels_dict["class"]
                groups = labels_dict["group"]
                places = labels_dict["spurious"]
                predicted = labels_dict["ebd_y_pred"]

                total_labels.extend(labels.numpy())
                total_groups.extend(groups.numpy())
                total_spuriouss.extend(places.numpy())
                total_predictions.extend(predicted.numpy())
                total_embeddings.extend(embeddings.numpy())
                
                del embeddings; del labels; del groups; del places; del predicted

        total_embeddings = np.array(total_embeddings) # (# of full data, feat_dim)

        total_meta_results = {"targets" : total_labels, "spuriouss": total_spuriouss, "groups" : total_groups, 
                             "predictions": total_predictions}
        
        return total_embeddings, total_meta_results
        
    def VisRep(self, model, dataloader, vis_on, label_types=['group', 'target', 'spurious', 'prediction'], num_data=None, reduced_dim=2,
                          figsize=(8, 6), save=True, ftype='.png', title_suffix=None, save_id_suffix=None,
                          annotate_points=None, plot_mds=False, seed=42):
        """_summary_

        Args:
            model (_type_): nn.Module
            dataloader (_type_): Dataset for visualization 
            vis_on (_type_): choice <- ["train", "val", "test", "val_fg", "test_fg"] (correspond to dataloader)
            label_types (_type_): ['confidence', 'target', 'spurious', 'group', 'prediction']
            num_data (_type_, optional): for Random sampling(No..) Defaults to None=Full..
            reduced_dim (int, optional): _description_. Defaults to 2.
        """
        
        total_embeddings, total_meta_results = self.GetEmbeddings(model, dataloader)
        
        if self.args.tl_method == "linear_probing":
            model_title = "CLIP ZS"
            title_suffix= f'([{model_title}] Rep. on [{vis_on}])'
        else:
            title_suffix= f'([{self.args.tl_method}] Rep. on [{vis_on}] (Epoch {self.model_epoch}))'
            
             

        print(f'total_embeddings.shape: {total_embeddings.shape}')
        n_mult = 1
        pbar = tqdm(total=n_mult * len(label_types))
        for label_type in label_types:
            # For now just save both classifier ReLU activation layers (for MLP, BaseCNN)
            if save_id_suffix is not None:
                save_id = f'{reduced_dim}d_{label_type}_{vis_on}_{save_id_suffix}'
            else:
                save_id = f'{reduced_dim}d_{label_type}_{vis_on}'
                
            plot_umap(total_embeddings, total_meta_results, label_type, self.legend_labels_dict, reduced_dim, num_data, method='umap',
                        offset=0, figsize=figsize, save_id=save_id, save=save,
                        ftype=ftype, title_suffix=title_suffix, annotate_points=annotate_points,
                        seed=seed, display_image = True)
            # Add MDS
            if plot_mds:
                plot_umap(total_embeddings, total_meta_results, label_type, self.legend_labels_dict,  reduced_dim, num_data, method='mds',
                            offset=0, figsize=figsize, save_id=save_id, save=save,
                            ftype=ftype, title_suffix=title_suffix, annotate_points=annotate_points,
                            seed=seed, display_image = True)
            pbar.update(1)
    
    def VisRepAll(self, train_loader, val_loader, test_loader, label_types=['group', 'target', 'spurious', 'prediction'], num_data=None, reduced_dim=2,
                          figsize=(24, 6), save=True, ftype='.png', title_suffix=None, save_id_suffix=None,
                          annotate_points=None, plot_mds=False, seed=42, text_ebd=None, group_mean_ebd=None, num_nn_text_ebd=10, set_bbox=False):
        """
        - Projection all train/val/test sets to same sub-space. (thus same umap-structure)
        """
        
        # self.embeddings_df = pd.read_json(self.embedding_dir) # key : image_filename
        indices_to_convert = ['y', 'place', 'group', 'y_pred', 'split'] # str -> int
        # self.embeddings_df.loc[indices_to_convert] = self.embeddings_df.loc[indices_to_convert].astype('int64')
        
        total_embeddings_train, total_meta_results_train = self.GetEmbeddings(train_loader)
        total_embeddings_val, total_meta_results_val = self.GetEmbeddings(val_loader)
        total_embeddings_test, total_meta_results_test = self.GetEmbeddings(test_loader)
        
        # Save Group-wise Statistics -> [(norm of mean_vector, mean-vector) / compactness] for [train/val/test]
        
        print("> Calculating [Group-wise] Statistics...")
        self.group_wise_stat_ebd['train'] = GetGroupWiseStatEbd(total_embeddings_train, np.array(total_meta_results_train["groups"]))
        self.group_wise_stat_ebd['val'] = GetGroupWiseStatEbd(total_embeddings_val, np.array(total_meta_results_val["groups"]))
        self.group_wise_stat_ebd['test'] = GetGroupWiseStatEbd(total_embeddings_test, np.array(total_meta_results_test["groups"]))
            
        group_wise_indexes = ["Acc.", "Div.", "Centr. Norm."]
        columns = ["Avg.","Worst", "group0", "group1", "group2", "group3"]
        
        dfs =[]
        for split in ["train", "val", "test"]:
            if split=="train":
                values = [list(self.zs_results[f"{split}"].values())[:-1], 
                        [list(self.group_wise_stat_ebd[split]["pairwise_distance"].values())[0]] + [0] + list(self.group_wise_stat_ebd[split]["pairwise_distance"].values())[1:],
                        [list(self.group_wise_stat_ebd[split]["mean_vector_norm"].values())[0]] + [0] + list(self.group_wise_stat_ebd[split]["mean_vector_norm"].values())[1:]]
                df = pd.DataFrame(values, index=group_wise_indexes, columns = columns)
                df = df.round(3)
                dfs.append(df)
            else:
                values = [list(self.zs_results[f"{split}"].values())[:-1], 
                        [list(self.group_wise_stat_ebd[split]["pairwise_distance"].values())[0]] + [0] + list(self.group_wise_stat_ebd[split]["pairwise_distance"].values())[1:],
                        [list(self.group_wise_stat_ebd[split]["mean_vector_norm"].values())[0]] + [0] + list(self.group_wise_stat_ebd[split]["mean_vector_norm"].values())[1:]]
                df = pd.DataFrame(values, index=group_wise_indexes, columns = columns)
                df = df.round(3)
                dfs.append(df)
        
        if group_mean_ebd is not None:  # group label : (4, 2024) X 3
            add_group_labels_train = [group for group in self.group_wise_stat_ebd['train']["mean_vector"].keys()] # Waterbird
            add_group_mean_ebds_train = [ebd for ebd in self.group_wise_stat_ebd['train']["mean_vector"].values()]
            add_group_labels_val = [group for group in self.group_wise_stat_ebd['val']["mean_vector"].keys()] # Waterbird
            add_group_mean_ebds_val = [ebd for ebd in self.group_wise_stat_ebd['val']["mean_vector"].values()]
            add_group_labels_test = [group for group in self.group_wise_stat_ebd['test']["mean_vector"].keys()] # Waterbird
            add_group_mean_ebds_test = [ebd for ebd in self.group_wise_stat_ebd['test']["mean_vector"].values()]

            group_mean_ebd = (add_group_mean_ebds_train, add_group_mean_ebds_val, add_group_mean_ebds_test,
                              add_group_labels_train, add_group_labels_val, add_group_labels_test)
        
        
        
        if self.args.tl_method == "linear_probing":
            title_suffix= f'([CLIP ZS] Representation ({num_nn_text_ebd} near.)'
        else:
            title_suffix= f'([{self.args.tl_method}] Representation ({num_nn_text_ebd} near.) (Epoch {self.model_epoch}))'
        
        n_mult = 1
        pbar = tqdm(total=n_mult * len(label_types))
        for label_type in label_types:
            # For now just save both classifier ReLU activation layers (for MLP, BaseCNN)
            if save_id_suffix is not None:
                save_id = f'{reduced_dim}d_{label_type}_{save_id_suffix}'
            else:
                save_id = f'{reduced_dim}d_{label_type}'
            
            print("save_id", save_id)
            
            plot_umap_all(total_embeddings_train, total_embeddings_val, total_embeddings_test, total_meta_results_train, total_meta_results_val, total_meta_results_test,
                  label_type, self.legend_labels_dict, dfs, reduced_dim, method='umap', figsize=figsize, save_id=save_id, save=save, ftype=ftype, title_suffix=title_suffix,
              annotate_points=annotate_points, seed=seed, display_image=True, text_ebd = text_ebd, group_mean_ebd = group_mean_ebd, num_nn_text_ebd = num_nn_text_ebd, set_bbox=set_bbox)
            
            if plot_mds:
                plot_umap_all(total_embeddings_train, total_embeddings_val, total_embeddings_test, total_meta_results_train, total_meta_results_val, total_meta_results_test,
                  label_type, self.legend_labels_dict, dfs, reduced_dim, method='mds', figsize=figsize, save_id=save_id, save=save, ftype=ftype, title_suffix=title_suffix,
              annotate_points=annotate_points, seed=seed, display_image=True, text_ebd = text_ebd, group_mean_ebd = group_mean_ebd, num_nn_text_ebd = num_nn_text_ebd, set_bbox=set_bbox)
            
            pbar.update(1)

# Utils
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def read_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def print_param_change(old_model, new_model):
    """
    old_model : torch.nn.module
    new_model : torch.nn.module
    """
    for (name, parameter), (name_, parameter_) in zip(old_model.named_parameters(), new_model.named_parameters()):
        param_shape = parameter.shape
        final_value = parameter[tuple([idx-1 for idx in param_shape])]
        final_value_ = parameter_[tuple([idx-1 for idx in param_shape])]
        
        print(f"{name}: {final_value} -> {final_value_}")
        
class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        try:
            module_out = module_out.detach().cpu()
            self.outputs.append(module_out)  # .detach().cpu().numpy()
        except Exception as e:
            print(e)
            self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


def ChangeDictKeyOrder(old_dict, new_key_order, round=None):
    
    if round is None: 
        new_dict =  {key: old_dict[key] for key in new_key_order}
    else:
        new_dict =  {key: np.round(old_dict[key], round) for key in new_key_order}
    return new_dict

def plot_umap(embeddings, meta_results,  label_type, legend_labels_dict, reduced_dim=2, num_data=None, method='umap',
              offset=0, figsize=(12, 9), save_id=None, save=True,
              ftype='.png', title_suffix=None, annotate_points=None, seed=42, display_image=True):
    """
    Visualize embeddings with U-MAP
    - embeddings : embeddings
    - meta_results : corresponding results > dict_keys(['labels', 'spurioiuss', 'groups', 'confidences', 'predictions'])
    """
    labels = np.array(meta_results[label_type+'s'])
    
    if num_data is None:
        embeddings = embeddings
    elif offset == 0:
        np.random.seed(seed)
        num_data = np.min((num_data, len(embeddings)))
        sample_ix = np.random.choice(np.arange(len(embeddings)),
                                     size=num_data, replace=False)

        embeddings = embeddings[sample_ix]
        
        labels = labels[sample_ix]
    else:
        embeddings = embeddings[offset:offset + num_data]
        labels = labels[offset:offset + num_data]
    
    if label_type == 'confidence':
        colors = np.array(labels)
    else:    
        colors = np.array(labels).astype(int)
        num_colors = len(np.unique(colors))
        if num_colors==2:
            colors_template = ['midnightblue', 'red']
        elif num_colors==4: # Group
            colors_template = ['midnightblue', 'darkorange', 'red', 'royalblue']
        colors = [colors_template[val] for val in np.array(labels)] 
    
    fig = plt.figure(figsize=figsize)

    # zero_embedding = np.zeros_like(embeddings[0].shape)
    
    if method == 'umap':
        standard_embedding = umap.UMAP(random_state=42, n_components=reduced_dim).fit_transform(embeddings)
    else:  # method == 'mds'
        standard_embedding = MDS(n_components=reduced_dim,
                                 random_state=42).fit_transform(embeddings)
    
    if reduced_dim==2:
        # Continuous
        if label_type == 'confidence':
            scatter = plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1],
                    c=colors, s=1.0, alpha=1,
                    cmap=plt.cm.get_cmap('coolwarm'))
        # Discrete
        else:
            plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=colors, s=1.0, alpha=1)

    else:
        assert reduced_dim==3
        
        ax = fig.add_subplot(111, projection='3d')
        
        # Continuous
        if label_type == 'confidence':
            scatter = ax.scatter(standard_embedding[:, 0], standard_embedding[:, 1],  standard_embedding[:, 2],
                    c=colors, s=1.0, alpha=1,
                    cmap=plt.cm.get_cmap('coolwarm'))
        
        # Discrete
        else:
            ax.scatter(standard_embedding[:, 0], standard_embedding[:, 1],  standard_embedding[:, 2],
                    c=colors, s=1.0, alpha=1)
            
    if label_type == 'confidence':
        cbar = plt.colorbar(scatter)
        cbar.set_label('Confidence')
    else:
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors_template]
        legend_labels = [legend_labels_dict[label_type][label] for label in range(len(np.unique(labels)))]
        if reduced_dim==2:
            plt.legend(legend_elements, legend_labels)
        else:
            assert reduced_dim==3
            ax.legend(legend_elements, legend_labels)
    
    
            
    suffix = '' if title_suffix is None else f' {title_suffix}'
    plt.title(f'Color by [{label_type}] labels{suffix}')
    
    if save:
        fpath = f'{save_id}{ftype}'
        plt.savefig(fname=fpath, dpi=300, bbox_inches="tight")
        print(f'Saved {method} to {fpath}!')
        
    if display_image:
        plt.show()
    plt.close('all')
    del standard_embedding
    

def plot_umap_all(embeddings_train, embeddings_val, embeddings_test, meta_results_train, meta_results_val, meta_results_test,
                  label_type, legend_labels_dict, passed_dfs, reduced_dim=2,  method='umap', figsize=(24, 6), save_id=None, save=True,
              ftype='.png', title_suffix=None, annotate_points=None, seed=42, display_image=True, text_ebd = None, group_mean_ebd = None, num_nn_text_ebd = 10, remove_prefix=True, set_bbox=False):
    """
    Visualize embeddings with U-MAP
    - embeddings : embeddings
    - meta_results : corresponding results > dict_keys(['labels', 'spurioiuss', 'groups', 'confidences', 'predictions'])
    """

    labels_train = np.array(meta_results_train[label_type+'s'])
    labels_val = np.array(meta_results_val[label_type+'s'])
    labels_test = np.array(meta_results_test[label_type+'s'])
    
    n_train = len(labels_train)
    n_val = len(labels_val)
    n_test = len(labels_test)
    
    print(f"Number of dataset : tr : [{n_train}], val : [{n_val}], test : [{n_test}]")
    total_labels = np.concatenate([labels_train, labels_val, labels_test], axis=0)
    total_embeddings = np.concatenate([embeddings_train, embeddings_val, embeddings_test], axis=0)
    print("ㄴ total embedddings  : ", total_embeddings.shape)

    # if (text_ebd is not None) or (group_mean_ebd is not None):
    #     print("Add [zero] embedding to umap-pool")
    #     zero_ebd = np.zeros((1, total_embeddings[0].shape[0]))
    #     total_embeddings = np.concatenate([total_embeddings, zero_ebd], axis=0)
    #     print("ㄴ total embedddings  : ", total_embeddings.shape)

    if text_ebd is not None: # (# of templates, 2048) add to embedding pool
        print("Add [text] embedding to umap-pool")
        text_templates = [list(temp_feat_pair.keys())[0] for temp_feat_pair in text_ebd]
        text_features = [list(temp_feat_pair.values())[0] for temp_feat_pair in text_ebd]
        
        print(f"> Calculate {num_nn_text_ebd} Nearest samples for visualization of [text prompts]")
        nearest_averaged_text_features = []
        for i in range(len(text_features)):
            nearest_indices = find_closest_sample(total_embeddings, text_features[i], top_k=num_nn_text_ebd)
            nearest_averaged_embedding = total_embeddings[nearest_indices].mean(axis=0)
            nearest_averaged_text_features.append(nearest_averaged_embedding)
        
        # # Scale text-embeddings(12.xx) to image-scale (2.xx)
        # [print(compute_vector_norm(feat)) for feat in text_features]
        # norm_full_image = compute_vector_norm(total_embeddings.mean(axis=0))
        # text_features = [(text_feat / compute_vector_norm(text_feat))*norm_full_image for text_feat in text_features]
        # [print(compute_vector_norm(feat)) for feat in text_features]
        
        total_embeddings = np.concatenate([total_embeddings, np.array(nearest_averaged_text_features)], axis=0)
        print("ㄴ total embedddings  : ", total_embeddings.shape)
        # Label : 0, 1
    
    if  group_mean_ebd is not None:
        # 각각 (4, 2024), (4, 2024), (4, 2024)
        # label : 0, 1 ,2, 3
        print("Add [group] (mean) embedding to umap-pool")
        (add_group_mean_ebds_train, add_group_mean_ebds_val, add_group_mean_ebds_test,
                              add_group_labels_train, add_group_labels_val, add_group_labels_test) = group_mean_ebd 
        
        add_group_mean_ebds = np.concatenate([add_group_mean_ebds_train, add_group_mean_ebds_val, add_group_mean_ebds_test], axis=0)
        total_embeddings = np.concatenate([total_embeddings, add_group_mean_ebds])
        
        print("ㄴ total embedddings  : ", total_embeddings.shape)
    
    print("> Projection all the embeddings to [1024d l2-norm sphere]")
    total_embeddings = total_embeddings / np.linalg.norm(total_embeddings, axis=1, keepdims=True)
    print(f"> Start Umap fitting.... (# of samples {total_embeddings.shape[0]})(dim {total_embeddings.shape[1]})")
    if method == 'umap':
        standard_embedding = umap.UMAP(random_state=42, n_components=reduced_dim).fit_transform(total_embeddings)
    else:  # method == 'mds'
        standard_embedding = MDS(n_components=reduced_dim,
                                 random_state=42).fit_transform(total_embeddings)
    
    standard_embedding_train = standard_embedding[: n_train]
    standard_embedding_val = standard_embedding[n_train: n_train + n_val]
    standard_embedding_test = standard_embedding[n_train+n_val : n_train+n_val+n_test]
    
    offset_for_add = n_train+n_val+n_test
    # if (text_ebd is not None) or (group_mean_ebd is not None):
    #     standard_zero_ebd = standard_embedding[offset_for_add]
    #     offset_for_add = offset_for_add + 1 
    #     print("standard [zero] ebd' shape:", standard_zero_ebd.shape)
        
    if text_ebd is not None:
        offset_for_text_ebd = len(text_templates)
        standard_text_ebd = standard_embedding[offset_for_add: offset_for_add + offset_for_text_ebd]
        print("standard [text]] ebd' shape:", standard_text_ebd.shape)
        offset_for_add = offset_for_add + offset_for_text_ebd
    
    if group_mean_ebd is not None:
        standard_group_mean_ebd_train = standard_embedding[offset_for_add: offset_for_add + 5] # Mean + Group 4
        standard_group_mean_ebd_val = standard_embedding[offset_for_add + 5: offset_for_add + 10]
        standard_group_mean_ebd_test = standard_embedding[offset_for_add + 10: offset_for_add + 15]
        print("standard [group] ebd' shape:", standard_group_mean_ebd_test.shape)
                     
    fig = plt.figure(figsize=figsize)

    # Zero -> Text Prompt (원점 보정) -> CLIP에서는 안 되네.. 너무 Outlier인듯.
    standard_origin_ebd = standard_embedding.mean(axis=0)
    
    # standard_zero_ebd : all the ploting
    # standard_text_ebd : all the ploting 
    
    if reduced_dim == 2:
        fig, axs =plt.subplots(2,3, figsize=figsize,  gridspec_kw={'height_ratios': [2.5, 1]})
    
    else:
        assert reduced_dim ==3
        fig, axs =plt.subplots(2,3, figsize=figsize, subplot_kw={"projection": "3d"},  gridspec_kw={'height_ratios': [2.5, 1]})
        # fig, axs =plt.subplots(2,3, figsize=figsize,   gridspec_kw={'height_ratios': [3, 1]})
    for idx, (each_standard_embedding, labels, each_standard_group_mean_ebd, each_df, sub_title) in enumerate(zip([standard_embedding_train, standard_embedding_val, standard_embedding_test],
                                                                           [labels_train, labels_val, labels_test],
                                                                           [standard_group_mean_ebd_train, standard_group_mean_ebd_val, standard_group_mean_ebd_test],
                                                                           passed_dfs,
                                                                           ["Train set", "Val set", "Test set"])):
        # Group : train/val/test
        if label_type == 'confidence':
            colors = np.array(labels)
        else:    
            colors = np.array(labels).astype(int)
            num_colors = len(np.unique(colors))
            if num_colors==2:
                colors_template = ['midnightblue', 'red']
            elif num_colors==4: # Group
                colors_template = ['midnightblue', 'darkorange', 'red', 'royalblue']
            colors = [colors_template[val] for val in np.array(labels)] 
        
        if reduced_dim==2:
            # ax = fig.add_subplot(2, 3, idx+1)
            # Continuous
            if label_type == 'confidence':
                scatter = axs[0][idx].scatter(each_standard_embedding[:, 0], each_standard_embedding[:, 1],
                        c=colors, s=1.0, alpha=1,
                        cmap=plt.cm.get_cmap('coolwarm'))
            # Discrete
            else:
                axs[0][idx].scatter(each_standard_embedding[:, 0], each_standard_embedding[:, 1], c=colors, s=1.0, alpha=1)
            
            # print("X:", each_standard_embedding[0, 0])
            # print("Y:", each_standard_embedding[0, 1])
            # axs[0][0].annotate("Test~~", xytext=each_standard_embedding[0], xy=standard_origin_ebd, arrowprops=dict(arrowstyle="<-"), size=30)
            if text_ebd is not None:
                for idx_, ebd in enumerate(standard_text_ebd):
                    if remove_prefix:
                        if set_bbox:
                            axs[0][idx].annotate(f'"{text_templates[idx_].split("a photo of ")[-1]}"', xytext=ebd, xy=standard_origin_ebd, arrowprops=dict(arrowstyle="<|-"), bbox=dict(boxstyle="round4", fc="w"))
                        else:
                            axs[0][idx].annotate(f'"{text_templates[idx_].split("a photo of ")[-1]}"', xytext=ebd, xy=standard_origin_ebd, arrowprops=dict(arrowstyle="<|-") ) # bbox=dict(boxstyle="round4", fc="w")
                    else:
                        if set_bbox:
                            axs[0][idx].annotate(f'"{text_templates[idx_]}"', xytext=ebd, xy=standard_origin_ebd, arrowprops=dict(arrowstyle="<|-"), bbox=dict(boxstyle="round4", fc="w"))
                        else:
                            axs[0][idx].annotate(f'"{text_templates[idx_]}"', xytext=ebd, xy=standard_origin_ebd, arrowprops=dict(arrowstyle="<|-")) # bbox=dict(boxstyle="round4", fc="w")
            if group_mean_ebd is not None:
                for idx_, ebd in enumerate(each_standard_group_mean_ebd):
                    if idx_ ==0:
                        continue # Pass the average vector.
                    axs[0][idx].annotate(f"Group {idx_-1}", xytext=ebd, xy=standard_origin_ebd, arrowprops=dict(arrowstyle="<-"))
            # ax = fig.add_subplot(2, 3, idx+4)
            axs[1][idx].axis('tight')
            axs[1][idx].axis('off')
            table = axs[1][idx].table(cellText=each_df.values, colLabels=each_df.columns, rowLabels=each_df.index, loc='center')
            # ax.set_box_aspect(1)
        else:
            assert reduced_dim==3
            
            
            # axs[0][idx] = fig.add_subplot(2, 3, idx+1, projection='3d')
            
            # Continuous
            if label_type == 'confidence':
                scatter = axs[0][idx].scatter(each_standard_embedding[:, 0], each_standard_embedding[:, 1],  each_standard_embedding[:, 2],
                        c=colors, s=1.0, alpha=1,
                        cmap=plt.cm.get_cmap('coolwarm'))
            
            # Discrete
            else:
                axs[0][idx].scatter(each_standard_embedding[:, 0], each_standard_embedding[:, 1],  each_standard_embedding[:, 2],
                        c=colors, s=1.0, alpha=1)
            
            if text_ebd is not None:
                for idx_, ebd in enumerate(standard_text_ebd):
                    axs[0][idx].text(ebd[0],ebd[1],ebd[2], f"Prompt {text_templates[idx_]}", size=8)
                    axs[0][idx].arrow3D(standard_origin_ebd[0],standard_origin_ebd[1],standard_origin_ebd[2], ebd[0]-standard_origin_ebd[0],ebd[1]-standard_origin_ebd[1],ebd[2]-standard_origin_ebd[2], mutation_scale=20, arrowstyle="-|>", fc='red')
            if group_mean_ebd is not None:
                for idx_, ebd in enumerate(each_standard_group_mean_ebd):
                    if idx_ ==0:
                        continue # Pass the average vector.
                    
                    # axs[0][idx].annotate(f"Group {idx}", xytext=ebd, xy=standard_origin_ebd, arrowprops=dict(arrowstyle="<-"))
                    axs[0][idx].text(ebd[0],ebd[1],ebd[2], f"Group {idx_}", size=8)
                    # print(standard_origin_ebd[0],standard_origin_ebd[1],standard_origin_ebd[2], ebd[0],ebd[1],ebd[2])                    
                    axs[0][idx].arrow3D(standard_origin_ebd[0],standard_origin_ebd[1],standard_origin_ebd[2], ebd[0]-standard_origin_ebd[0],ebd[1]-standard_origin_ebd[1],ebd[2]-standard_origin_ebd[2], mutation_scale=20, arrowstyle="-|>", linestyle='dashed')
            
            
            axs[1][idx].axis('tight')
            axs[1][idx].axis('off')
            # ax = fig.add_subplot(2, 3, idx+4)
            table = axs[1][idx].table(cellText=each_df.values, colLabels=each_df.columns, rowLabels=each_df.index, loc='center')
            axs[1][idx].set_box_aspect([1, 1, 1])
            
        table.scale(1, 2)  # Adjust the scale factors to control the size of the table (매커니즘 몰라ㅏ)
        
            
        if label_type == 'confidence':
            cbar = plt.colorbar(scatter)
            cbar.set_label('Confidence')
        else:
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors_template]
            legend_labels = [legend_labels_dict[label_type][label] for label in range(len(np.unique(labels)))]
            axs[0][idx].legend(legend_elements, legend_labels)
            
        axs[0][idx].set_title(sub_title)
                
    suffix = '' if title_suffix is None else f' {title_suffix}'
    plt.suptitle(f'Color by [{label_type}] labels{suffix}', size=20)
    # plt.tight_layout(rect=[0, 0, 1, 0.95]) 
        
    if save:
        fpath = f'{save_id}{ftype}'
        if not os.path.exists("figure"):
            os.mkdir("figure")
        fpath = os.path.join("figure", fpath)
        plt.savefig(fname=fpath, dpi=300, bbox_inches="tight")
        print(f'Saved {method} to {fpath}!')
        
    if display_image:
        plt.show()
    plt.close('all')
    del standard_embedding
    
    
from scipy.spatial.distance import cdist
def compute_mean_vector(array, axis=0):
    mean_vector = np.mean(array, axis=axis)
    return mean_vector

def compute_vector_norm(vector):
    norm = np.linalg.norm(vector)
    return norm

def compute_averaged_pairwise_distance(array):
    num_samples = array.shape[0]
    pairwise_distances = cdist(array, array)
    average_distance = np.sum(pairwise_distances) / (num_samples * (num_samples - 1))
    return average_distance

# Measures : norm, mean_vector, compactness
def GetGroupWiseStatEbd(embeddings, group_labels):
    # in Each group
    statistics = {
        'mean_vector' : {},
        'mean_vector_norm' : {},
        'pairwise_distance' : {},
    }
    
    # Full datasets
    mean_vector = compute_mean_vector(embeddings, axis=0)
    vector_norm = compute_vector_norm(mean_vector)
    pairwise_distance = compute_averaged_pairwise_distance(embeddings)
    
    statistics['mean_vector']['full'] = mean_vector
    statistics['mean_vector_norm']['full'] = vector_norm
    statistics['pairwise_distance']['full'] = pairwise_distance

    for group in np.unique(group_labels): # 0, 1, 2, 3
        group_indices = np.where(group_labels == group)[0]
        group_embeddings = embeddings[group_indices]
        
        mean_vector = compute_mean_vector(group_embeddings, axis=0)
        vector_norm = compute_vector_norm(mean_vector)
        pairwise_distance = compute_averaged_pairwise_distance(group_embeddings)
        
        statistics['mean_vector'][group] = mean_vector
        statistics['mean_vector_norm'][group] = vector_norm
        statistics['pairwise_distance'][group] = pairwise_distance
    
    return statistics

def GetGroupWiseStatConf(confidences, group_labels):

    # in Each group
    statistics = {}
    # Full datasets

    mean_conf = np.mean(confidences)
    statistics['full'] = mean_conf

    for group in np.unique(group_labels): # 0, 1, 2, 3
        group_indices = np.where(group_labels == group)[0]
        group_confidences = confidences[group_indices]

        mean_conf = np.mean(group_confidences)
        statistics[group] = mean_conf
    
    return statistics

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D

from matplotlib.text import Annotation

class Annotation3D(Annotation):

    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)
        
def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add anotation `text` to an `Axes3d` instance.'''

    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)

setattr(Axes3D, 'annotate3D', _annotate3D)

import numpy as np
from matplotlib.patches import FancyArrowPatch

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)