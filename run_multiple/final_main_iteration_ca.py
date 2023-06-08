from __future__ import print_function
import numpy as np
import pandas as pd
import os
import sys
import json

# sys.path.append("/home/jinsu/workstation/project/debiasing-multi-modal") # when running in not-root folder 
import torch
import torch.nn as nn

import sys
import argparse
import time
import tqdm
import math
from copy import deepcopy


import torch
import torch.backends.cudnn as cudnn

from demo.util import AverageMeter
from demo.util import adjust_learning_rate, warmup_learning_rate, accuracy, adjust_learning_rate_reg, warmup_learning_rate_reg
from demo.util import set_optimizer, set_optimizer_reg, get_lr
from demo.util import set_seed
from visualizer_supcon import skim_dataloader_by_group # 임시, Balanced Loader 확인용.

from torch.utils.data import DataLoader, Subset
from data.waterbirds_embeddings import WaterbirdsEmbeddings, load_waterbirds_embeddings
from data.celeba_embeddings import CelebaEmbeddings, load_celeba_embeddings
print(CelebaEmbeddings)
model_dict = {'resnet50': [None, 1024]} # (nn.module, 1024)
new_order_for_print = [
    'weighted_mean_acc',
    'worst_acc',
    'acc_0_0',
    'acc_0_1',
    'acc_1_0',
    'acc_1_1',
    'mean_acc'
]
from functools import partial

class LinearClassifier(nn.Module): # Linear probing
    def __init__(self, input_dim, num_classes=2):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, features):
        return self.fc(features)



class CustomCLIP(nn.Module): # Adapter / Contrastive Adapter
    def __init__(self, adapter, text_embedding_dir, text_spurious_embedding_dir, text_group_embedding_dir, temperature=0.01):
        super().__init__()
        self.text_embedding_dir = text_embedding_dir 
        self.text_spurious_embedding_dir = text_spurious_embedding_dir
        self.text_group_embedding_dir = text_group_embedding_dir #NOTE Joonwon Added
        self.adapter = adapter
        self.temperature = temperature # CA default : 0.01, B2T default : 0.02 (?) NOTE
        
        self.text_features = get_text_embedding(self.text_embedding_dir).cuda()
        self.n_cls = self.text_features.shape[0]
        self.text_spurious_features = get_text_embedding(self.text_spurious_embedding_dir).cuda()
        
    def forward(self, features, use_group=False): 
        image_features =  self.adapter(features) # Un-normalized (B, 1024)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # Normalized (B, 1024)

        #NOTE Joonwon Added
        if use_group:
            text_features = get_text_embedding(self.text_group_embedding_dir).cuda() # (Pre) Normalized (B, 2, 1024)
        else:
            text_features = self.text_features # (Pre) Normalized (B, 2, 1024)
        
        # Check if we have to normalize the text features
        text_features = text_features / text_features.norm(dim=0, keepdim=True)
        logits = image_features @ text_features / self.temperature # (B, 1024) X (B, C, 1024) = # (B, C)
        
        return logits
    
    def forward_spurious(self, features): 
        image_features =  self.adapter(features) # Un-normalized (B, 1024)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # Normalized (B, 1024)

        text_spurious_features = self.text_spurious_features # 
        text_spurious_features = text_spurious_features / text_spurious_features.norm(dim=0, keepdim=True) # 
        
        
        logits = image_features @ text_spurious_features / self.temperature # (B, 1024) X (1024, 2) = # (B, 2)
        
        return logits


# 제대로 복사 됐는지.
# 파라미터 Freezing(detach) 잘 되는지. 
class MultipleAdapter(nn.Module): # Adapter / Contrastive Adapter
    def __init__(self, old_cls, new_adapter, init_near_identity=True, ebd_weight=0.5):
        super().__init__()
        self.old_cls = old_cls
        self.text_embedding_dir = self.old_cls.text_embedding_dir # 따로 짜둔 코드에서 오류 안 생기게 전부 할당.
        self.text_spurious_embedding_dir = self.old_cls.text_spurious_embedding_dir
        self.text_group_embedding_dir = self.old_cls.text_group_embedding_dir #NOTE Joonwon Added
        
        self.text_features = get_text_embedding(self.text_embedding_dir).cuda()
        self.n_cls = self.text_features.shape[0]
        self.text_spurious_features = get_text_embedding(self.text_spurious_embedding_dir).cuda()
        
        
        self.new_adapter = new_adapter
        self.ebd_weight = ebd_weight
        if init_near_identity:
            print("Initialize paramters of [New adapter] from [Old adapter]")
            
            self.new_adapter.load_state_dict(self.old_cls.adapter.state_dict())
            
        self.temperature = self.old_cls.temperature
    
        
        
    def forward(self, features, use_group=False): 
        old_image_features =  self.old_cls.adapter(features) # Un-normalized (B, 1024)
        old_image_features = old_image_features / old_image_features.norm(dim=-1, keepdim=True) # Normalized (B, 1024)
        new_image_features = self.new_adapter(features) # Un-normalized (B, 1024)
        new_image_features = new_image_features / new_image_features.norm(dim=-1, keepdim=True) # Normalized (B, 1024)
        
        image_features = self.ebd_weight * old_image_features.detach() + (1 - self.ebd_weight) * new_image_features # 혹시 몰라 Detach까지.
        
        #NOTE Joonwon Added
        if use_group:
            text_features = get_text_embedding(self.text_group_embedding_dir).cuda() # (Pre) Normalized (B, 2, 1024)
        else:
            text_features = self.text_features # (Pre) Normalized (B, 2, 1024)
        
        # Check if we have to normalize the text features
        text_features = text_features / text_features.norm(dim=0, keepdim=True)
        
        logits = image_features @ text_features / self.temperature # (B, 1024) X (B, C, 1024) = # (B, C)
        
        return logits
    
    def forward_spurious(self, features): 
        old_image_features =  self.old_cls.adapter(features) # Un-normalized (B, 1024)
        old_image_features = old_image_features / old_image_features.norm(dim=-1, keepdim=True) # Normalized (B, 1024)
        
        new_image_features = self.new_adapter(features) # Un-normalized (B, 1024)
        new_image_features = new_image_features / new_image_features.norm(dim=-1, keepdim=True) # Normalized (B, 1024)
        
        image_features = self.ebd_weight * old_image_features.detach() + (1 - self.ebd_weight) * new_image_features # 혹시 몰라 Detach까지.
        
        
        text_spurious_features = self.text_spurious_features # 
        text_spurious_features = text_spurious_features / text_spurious_features.norm(dim=0, keepdim=True) # 
        

        logits = image_features @ text_spurious_features / self.temperature # (B, 1024) X (1024, 2) = # (B, 2)
        
        return logits

class Adapter(nn.Module):
    """
    - Residual connetion : 제외 (original Adapter - 0.2*images + 0.8*adapter)
    - Hidden dimension : args.adapter_feat_dim (original Adatper - input_dim // 4)
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, features):
        return self.layers(features)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--batch_size_reg', type=int, default=128,
                        help='batch_size for adpater_reg')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-1, 
                        help='learning rate')
    parser.add_argument('--learning_rate_reg', type=float, default=1e-3, 
                        help='learning rate for "adapter_reg_seq" model, in stage 2 ') 
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=1,
                        help='decay rate for learning rate') 
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='weight decay') # Tuning needed. 
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='waterbirds',
                        choices=['celeba', 'waterbirds'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing') # Tuning needed. 
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training') # Tuning needed. 
    parser.add_argument('--warm_reg', action='store_true',
                        help='warm-up for stable "adapter_reg_seq" training') # Tuning needed. 

    parser.add_argument('--image_embedding_dir', type=str,
                        help='extracted image embedding')
    parser.add_argument('--text_embedding_dir', type=str,
                        help='extracted text embedding')
    parser.add_argument('--text_group_embedding_dir', type=str,
                        help='extracted group embedding')
    parser.add_argument('--text_spurious_embedding_dir', type=str,
                        help='extracted text embedding (about spurious attributes)')
    parser.add_argument('--train_target', type=str, default="class", choices=["class", "spurious", "group"]) # label for prediction.
    parser.add_argument('--data_dir', type=str,
                    help='folder, in which metadata.csv] exist')
    parser.add_argument('--tl_method', type=str, default= "linear_probing", choices=["linear_probing", "adapter", "adapter_reg", "adapter_reg_seq","adapter_reg_seq_alter", "contrastive_adapter"]
                        ,help='transfer learning method')
    parser.add_argument('--balance_val', action='store_true', help="Balancing Val-reg loader.") #
    parser.add_argument('--resample_ce', action='store_true', help="Re-sampling Train loader.") #
    parser.add_argument('--use_cls_prompt_in_reg', action='store_true', help="True: use class-text-prompt in regularization.") # [10, 50] in Waterbird
    parser.add_argument('--add_adapter', action='store_true', default=False, help="Additional Adapter in regularization.") # [10, 50] in Waterbird
    parser.add_argument('--init_near_identity', action='store_true', help="Initialize additional adapter, making classifier' output near-identity before/after initialization ") # [10, 50] in Waterbird
    
    parser.add_argument('--epochs_feature_learning', type=int, help="epochs for feature learning in 'adapter_reg_seq'") # [10, 50] in Waterbird
    parser.add_argument('--continue_from_best', action='store_true', help="In stage 2, start from the best-worst-acc model.")
    parser.add_argument('--adapter_feat_dim', type=int, default= 128, help='reduced dimension in adapter')
    parser.add_argument('--zs_temperature', type=float, default= 0.01, help='Temperature in zero-shot prediction')
    parser.add_argument('--watch_batch_results', action='store_true', help='Print results in each bach by [opt.print_freq]. Recommdned: True when single-run of CelebA(Large # of batch), False others')
    parser.add_argument('--save_results', action='store_true', help='Save the results of transfer learning (and final feature quality) in the folder where ')
    
    parser.add_argument('--num_iter', type=int, default=3, help="Averaging [num_iter] run, at different seed")
    parser.add_argument('--random_seeds', type=str, default='42,32,22', help="random seed for iterative training." )
    parser.add_argument('--lr_multiple', type=float, default=1.0, help="lr multiple." )


    parser.add_argument('--lr_list', type=str, default='1e-1', help='Learning rate list')
    parser.add_argument('--bs_list', type=str, default='128', help='Batch size list')
    parser.add_argument('--bsr_list', type=str, default='128', help='Batch size (reg) list')


    # parser.add_argument('--lr_linear_probing', type=float, default=1e-3, chocies=[1e-3, 1e-2, 1e-1, 1, 3, 10], help='learning rate for linear probing') # Tuning needed. 
      # -> Zero-shot으로 대체하는 게 맞을듯.

    opt = parser.parse_args()


    # set the random seeds.
    random_seeds = opt.random_seeds.split(',')
    opt.random_seeds = [int(seed) for seed in random_seeds]
    print("random seeds : ", opt.random_seeds)
    
    # set the path according to the environment
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
    
    if opt.warm_reg:
        opt.warmup_from_reg = opt.learning_rate_reg / 1e2
        
        if opt.dataset =='celeba':
            opt.warm_epochs_reg = 2
        else:
            opt.warm_epochs_reg = 10
            
        if opt.cosine:
            eta_min = opt.learning_rate_reg * (opt.lr_decay_rate ** 3)
            opt.warmup_to_reg = eta_min + (opt.learning_rate_reg - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs_reg / (opt.epochs - opt.epochs_feature_learning))) / 2
        else:
            opt.warmup_to_reg = opt.learning_rate_reg
    
    if opt.dataset == 'celeba':
        opt.n_cls = 2
    elif opt.dataset == 'waterbirds':
        opt.n_cls = 2
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    
    if opt.tl_method == "adapter":
        assert not opt.add_adapter
        assert not opt.balance_val
    return opt


def set_model(opt):
    # model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()
        
    _ , input_dim = model_dict[opt.model] # (Encoder(not use), feature dim)
    
    if opt.tl_method =='linear_probing':
        print("Off-the-shelf classifier : [Linear Classifier]")
        classifier = LinearClassifier(input_dim = input_dim, num_classes = opt.n_cls)
    elif opt.tl_method =='adapter':
        print("Off-the-shelf classifier : [Adapter + (temperatured) image-text jointly normalized prediction]")
        adapter = Adapter(input_dim = input_dim, hidden_dim = opt.adapter_feat_dim) # Fixed by heuristics
        classifier = CustomCLIP(adapter, opt.text_embedding_dir, opt.text_spurious_embedding_dir, opt.text_group_embedding_dir, temperature=opt.zs_temperature)
    elif opt.tl_method in ['adapter_reg', 'adapter_reg_seq', 'adapter_reg_seq_alter']:
        print("Off-the-shelf classifier : [Adapter + (temperatured) image-text jointly normalized prediction] with group regularized training")
        adapter = Adapter(input_dim = input_dim, hidden_dim = opt.adapter_feat_dim) # Fixed by heuristics
        classifier = CustomCLIP(adapter, opt.text_embedding_dir, opt.text_spurious_embedding_dir, opt.text_group_embedding_dir, temperature=opt.zs_temperature)

    if torch.cuda.is_available():
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return classifier, criterion # model, 

def set_model_multiple_adapter(opt, erm_classifier):
    criterion = torch.nn.CrossEntropyLoss()
        
    _ , input_dim = model_dict[opt.model] # (Encoder(not use), feature dim)
    
    assert opt.tl_method in ['adapter_reg_seq', 'adapter_reg_seq_alter']
    
    print("================== Stage 2) New adapter for Balanced-Text-Prompt ==================")
    
    new_adapter = Adapter(input_dim = input_dim, hidden_dim = opt.adapter_feat_dim) # Fixed by heuristics
    
    new_classifier =  MultipleAdapter(erm_classifier, new_adapter, init_near_identity=opt.init_near_identity, ebd_weight=0.5) 
    
    if torch.cuda.is_available():
        classifier = new_classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return classifier, criterion # model, 


def balance_val(val_loader, opt, print_procedure = False):
    sub_dataset = val_loader.dataset
    n_groups = sub_dataset.dataset.n_groups
    g_idx = [np.where(sub_dataset.dataset.group_array[sub_dataset.indices] == g)[0] for g in range(n_groups)]
    min_g = np.min([len(g) for g in g_idx]) 
    
    if print_procedure:  print("> Before balancing (Shuffling for every epoch)")
    for i, g in enumerate(g_idx):
        np.random.shuffle(g) # 같은 Group 내에 있는 sample들 Shuffle (for up-sampling)
        if print_procedure:  print(f"(Group {i}): ", len(g), g[:10])
        
        # Balancing
        g_idx[i] = g[:min_g] 
        # print(f"After balancing (Group {i}): ", len(g_idx[i]), g_idx[i][:10])

    if print_procedure: print("> After balancing")
    [print(f"Group {i} : {len(g)} per epoch ({g[:4]})") for i, g in enumerate(g_idx) if print_procedure]
    balanced_indices = list(zip(*g_idx))

    balanced_indices = np.array(balanced_indices)
    balanced_indices = balanced_indices.reshape(-1)
    if print_procedure:  print(f"Balanced sample indices : {len(balanced_indices)} per epoch ({balanced_indices[:16]})")
    
    
    if opt.batch_size_reg <= len(balanced_indices):
        adjusted_batch_size_reg = opt.batch_size_reg
    else:
        if print_procedure: print(f"Adjust batch size for regularizaiton : [{opt.batch_size_reg}] -> [{len(balanced_indices)}]")
        adjusted_batch_size_reg = len(balanced_indices)
        
    balanced_subset = Subset(sub_dataset, balanced_indices)
    balanced_loader = DataLoader(balanced_subset, shuffle=False, batch_size = adjusted_batch_size_reg)
    
    return balanced_loader
            

# Group-wise Accuracy Update.
def update_dict(acc_groups, y, g, logits):
    preds = torch.argmax(logits, axis=1)
    correct_batch = (preds == y)
    g = g.cpu()
    for g_val in np.unique(g):
        mask = g == g_val
        n = mask.sum().item()
        corr = correct_batch[mask].sum().item()
        acc_groups[g_val].update(corr / n, n) 


# Mean/Worst acc (not weighted average)
def get_results(acc_groups, get_yp_func): # Input 중 acc_groups : AverageMeter()를 담고있는 dict. get_yp_func : 미리 partial을 이용해 n_groups를 저장해놓음. 
    groups = acc_groups.keys() # 0, 1, 2, 3
    results = {
            f"acc_{get_yp_func(g)[0]}_{get_yp_func(g)[1]}": acc_groups[g].avg
            for g in groups
    }
    all_correct = sum([acc_groups[g].sum for g in groups])
    all_total = sum([acc_groups[g].count for g in groups])
    results.update({"mean_acc" : all_correct / all_total})
    results.update({"worst_acc" : min(results.values())})
    
    return results

# Group -> class / spurious attributes
def get_y_p(g, n_places):
    y = g // n_places
    p = g % n_places
    return y, p

def get_text_embedding(text_embedding_dir):
    with open(text_embedding_dir, 'r') as f:
        text_embeddings = json.load(f)

    text_features = []
    for class_template, class_embedding in text_embeddings.items():
        text_features.append(torch.tensor(class_embedding))
    text_features = torch.stack(text_features, dim=1) # (B, 2, 1024)
    
    
    return text_features

def train_one_epoch(opt, train_loader, 
                    classifier, criterion, optimizer, epoch, get_yp_func, target, print_label='Train', predict_group = True): # model,
    """one epoch training"""
    # model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_groups = {g_idx : AverageMeter() for g_idx in range(train_loader.dataset.n_groups)}

    end = time.time()
    for idx, data in enumerate(train_loader):  
        
        embeddings, all_labels, img_filenames = data # all_labels.keys() : ['class', 'group', 'spurious', 'ebd_pred'(CLIP-zeroshot)] 
        labels = all_labels[target] # target : one of [y, spurious, group]
        groups = all_labels['group'] # For evaluating group accuracy (and further developing group-information-aware approaches)
    
        data_time.update(time.time() - end)

        embeddings = embeddings.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = classifier(embeddings.detach())  
        loss = criterion(output, labels) 

        # update metric
        losses.update(loss.item(), bsz)
        acc1 = accuracy(output, labels, bsz)
        acc.update(acc1, bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Update acc dict
        update_dict(acc_groups, labels, groups, output)
        
        group_acc = get_results(acc_groups, get_yp_func) # NOTE declared in [def main]
        group_acc = {key: group_acc[key] for key in new_order_for_print[1:]}
        group_acc = {key: np.round(value, 4) for key, value in group_acc.items()}
        
        if opt.watch_batch_results:
            if (idx + 1) % opt.print_freq == 0:
                print(f'{print_label}: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'Acc@1 {acc.val:.3f} ({acc.avg:.3f})\t'
                    'Group Acc {group_acc}'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, acc=acc, group_acc = group_acc))
                sys.stdout.flush()
            
    group_acc = get_results(acc_groups, get_yp_func) # NOTE declared in [def main]
    group_acc = {key: group_acc[key] for key in new_order_for_print[1:]}
    group_acc = {key: np.round(value, 4) for key, value in group_acc.items()}
    print(f"{print_label}:", str(group_acc))
    
    return losses.avg, acc.avg, group_acc

def train_reg_one_epoch(opt, train_loader1, train_loader2, classifier, criterion, optimizer, epoch, get_yp_func, target, group_prompt = True, print_label='Train'): # model,
    """one epoch training with regulalizar"""
    # model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_groups = {g_idx : AverageMeter() for g_idx in range(train_loader1.dataset.n_groups)}

    end = time.time()
    for dataloader, use_group in zip([train_loader1, train_loader2], [False, group_prompt]):

        for idx, data in enumerate(dataloader):  
            
            embeddings, all_labels, img_filenames = data # all_labels.keys() : ['class', 'group', 'spurious', 'ebd_pred'(CLIP-zeroshot)] 
            labels = all_labels[target] # target : one of [y, spurious, group]
            groups = all_labels['group'] # For evaluating group accuracy (and further developing group-information-aware approaches)
        
            data_time.update(time.time() - end)

            embeddings = embeddings.cuda(non_blocking=True)
            # NOTE joonwon added
            if use_group is True:
                labels = groups
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # warm-up learning rate
            warmup_learning_rate(opt, epoch, idx, len(dataloader), optimizer)

            # compute loss
            output = classifier(embeddings.detach(), use_group)  
            loss = criterion(output, labels) 

            # update metric
            if use_group is False:
                losses.update(loss.item(), bsz)
                acc1 = accuracy(output, labels, bsz)
                acc.update(acc1, bsz)

            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Update acc dict
            if use_group is False:
                update_dict(acc_groups, labels, groups, output)
            
            if opt.watch_batch_results:
                if (idx + 1) % opt.print_freq == 0:
                    print(f'{print_label}: [{0}][{1}/{2}]\t'
                        'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                        'Acc@1 {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch, idx + 1, len(dataloader), batch_time=batch_time,
                        data_time=data_time, loss=losses, acc=acc))
                    sys.stdout.flush()
                
    group_acc = get_results(acc_groups, get_yp_func) # NOTE declared in [def main]
    group_acc = {key: group_acc[key] for key in new_order_for_print[1:]}
    group_acc = {key: np.round(value, 4) for key, value in group_acc.items()}
    print(f"{print_label}:", str(group_acc))
    
    return losses.avg, acc.avg, group_acc

def train_reg_seq_one_epoch(opt, train_loader, classifier, criterion, optimizer, epoch, get_yp_func, target, print_label='Train', predict_group = True, use_group=False): # model,
    """one epoch training with regulalizar"""
    # model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    try: 
        acc_groups = {g_idx : AverageMeter() for g_idx in range(train_loader.dataset.n_groups)} # Train / Val / Test Loader
    except:
        try:
            acc_groups = {g_idx : AverageMeter() for g_idx in range(train_loader.dataset.dataset.n_groups)} # Val-reg / Val-eval Loadaer
        except:
            # ㅋㅋㅋ 
            acc_groups = {g_idx : AverageMeter() for g_idx in range(train_loader.dataset.dataset.dataset.n_groups)} # (Balanced) Val-reg Loader 

    end = time.time()
    
    for idx, data in enumerate(train_loader):  
        
        embeddings, all_labels, img_filenames = data # all_labels.keys() : ['class', 'group', 'spurious', 'ebd_pred'(CLIP-zeroshot)] 
        labels = all_labels[target] # target : one of [y, spurious, group]
        groups = all_labels['group'] # For evaluating group accuracy (and further developing group-information-aware approaches)
    
        data_time.update(time.time() - end)

        embeddings = embeddings.cuda(non_blocking=True)
        # NOTE joonwon added
        if use_group is True:
            labels = groups
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate_reg(opt, epoch - opt.epochs_feature_learning, idx, len(train_loader), optimizer)

        # compute loss
        output = classifier(embeddings.detach(), use_group)  
        loss = criterion(output, labels) 
        
        

        # update metric
        losses.update(loss.item(), bsz)
        acc1 = accuracy(output, labels, bsz)
        acc.update(acc1, bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Update acc dict
        update_dict(acc_groups, labels, groups, output)
        
        group_acc = get_results(acc_groups, get_yp_func) # NOTE declared in [def main]
        group_acc = {key: group_acc[key] for key in new_order_for_print[1:]}
        group_acc = {key: np.round(value, 4) for key, value in group_acc.items()}
        
        if opt.watch_batch_results:
            if (idx + 1) % opt.print_freq == 0:
                print(f'{print_label}: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'Acc@1 {acc.val:.3f} ({acc.avg:.3f})\t'
                    'Group Acc {group_acc}'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, acc=acc, group_acc = group_acc))
                sys.stdout.flush()
                
    group_acc = get_results(acc_groups, get_yp_func) # NOTE declared in [def main]
    group_acc = {key: group_acc[key] for key in new_order_for_print[1:]}
    group_acc = {key: np.round(value, 4) for key, value in group_acc.items()}
    print(f"{print_label}:", str(group_acc))
    
    return losses.avg, acc.avg, group_acc

def validate(opt, val_loader, classifier, criterion, get_yp_func, train_group_ratio, target, print_label='Test'):
    """validation"""
    
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    try:
        acc_groups = {g_idx : AverageMeter() for g_idx in range(val_loader.dataset.dataset.n_groups)}
    except:
        acc_groups = {g_idx : AverageMeter() for g_idx in range(val_loader.dataset.n_groups)}

    with torch.no_grad():
        end = time.time()
        for idx, data in enumerate(val_loader):
            embeddings, all_labels, img_filenames = data # all_labels.keys() : ['class', 'group', 'spurious', 'ebd_pred'(CLIP-zeroshot)] 
            labels = all_labels[target] # target : one of [class, spurious, group]
            groups = all_labels['group'] # For evaluating group accuracy (and further developing group-information-aware approaches)
            
            embeddings = embeddings.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(embeddings)
            loss = criterion(output, labels) #NOTE 준원 : validation은 그대로 두는 게 맞을듯 (class에 대해서 Training도 들어가고, 거기에 Validation 해야되는 거니까.)

            # update metric
            losses.update(loss.item(), bsz)
            acc1 = accuracy(output, labels, bsz)
            acc.update(acc1, bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Update acc dict
            update_dict(acc_groups, labels, groups, output)
        
            # if opt.watch_batch_results:
            #     if (idx+1) % opt.print_freq == 0:
            #         print(f'{print_label}: [{0}/{1}]\t'
            #             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #             'Acc@1 {acc.val:.3f} ({acc.avg:.3f})'.format(
            #             idx, len(val_loader), batch_time=batch_time,
            #             loss=losses, acc=acc))
                    
    group_acc = get_results(acc_groups, get_yp_func)

    # NOTE Add Weighted mean acc.
    try:
        groups = range(val_loader.dataset.dataset.n_groups) # 0, 1, 2, 3
    except:
        groups = range(val_loader.dataset.n_groups) # 0, 1, 2, 3
    group_acc_indiv =  [group_acc[f"acc_{get_yp_func(g)[0]}_{get_yp_func(g)[1]}"] for g in groups]
    weighted_mean_acc = (np.array(group_acc_indiv) * np.array(train_group_ratio)).sum() # Weighted Sum \
    
    group_acc["weighted_mean_acc"] = weighted_mean_acc
    group_acc = {key: group_acc[key] for key in new_order_for_print}
    group_acc = {key: np.round(value, 4) for key, value in group_acc.items()}
    print(f"{print_label}:", str(group_acc))

    return losses.avg, acc.avg, group_acc    



    

def validate_zs(opt, val_loader, classifier, criterion, get_yp_func, train_group_ratio, target, print_label='Zero-shot Prediction (Test) (Class)'):
    """(Feature quality) validation using zeroshot-prediction"""

    classifier.eval()

    if opt.tl_method in ["linear_probing"]:
        temperature = opt.zs_temperature
        
        if target=="class":
            text_embeddings = get_text_embedding(opt.text_embedding_dir)
        elif target=='spurious':
            text_embeddings = get_text_embedding(opt.text_spurious_embedding_dir)
        text_embeddings = text_embeddings.cuda()
        text_embeddings = text_embeddings / text_embeddings.norm(dim=0, keepdim=True)
        
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_groups = {g_idx : AverageMeter() for g_idx in range(val_loader.dataset.n_groups)}

    with torch.no_grad():
        end = time.time()
        for idx, data in enumerate(val_loader):
            image_embeddings, all_labels, img_filenames = data # all_labels.keys() : ['class', 'group', 'spurious', 'ebd_pred'(CLIP-zeroshot)] 
            labels = all_labels[target] # target : one of [class, spurious, group]
            groups = all_labels['group'] # For evaluating group accuracy (and further developing group-information-aware approaches)
            
            
            image_embeddings = image_embeddings.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            
            if opt.tl_method in ['linear_probing']: # same to CLIP Embedding
                image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True) # Normalized (B, 1024)
                output = image_embeddings @ text_embeddings / temperature # (B, 1024) X (B, 2, 1024) = # (B, 2)
                
            elif opt.tl_method in ['adapter', 'adapter_reg', 'adapter_reg_seq', 'adapter_reg_seq_alter', 'contrastive_adapter']: # Adpater, Contrastive Adapter : Embedding -> (1) (Adapted) Embedding -> (2) ZeroShot prediction as logit    (CustomCLIP.forward : (1)+(2))
                # forward
                if target=='class':
                    output = classifier(image_embeddings)
                elif target=='spurious':
                    output = classifier.forward_spurious(image_embeddings)
            
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1 = accuracy(output, labels, bsz)
            acc.update(acc1, bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Update acc dict
            update_dict(acc_groups, labels, groups, output)
        
            # if opt.watch_batch_results:
            #     if (idx+1) % opt.print_freq == 0:
            #         print(f'{print_label}: [{0}/{1}]\t'
            #             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #             'Acc@1 {acc.val:.3f} ({acc.avg:.3f})'.format(
            #             idx, len(val_loader), batch_time=batch_time,
            #             loss=losses, acc=acc))
                    
    group_acc = get_results(acc_groups, get_yp_func)

    # NOTE Add Weighted mean acc.
    groups = range(val_loader.dataset.n_groups) # 0, 1, 2, 3
    group_acc_indiv =  [group_acc[f"acc_{get_yp_func(g)[0]}_{get_yp_func(g)[1]}"] for g in groups]
    weighted_mean_acc = (np.array(group_acc_indiv) * np.array(train_group_ratio)).sum() # Weighted Sum \
    
    group_acc["weighted_mean_acc"] = weighted_mean_acc
    group_acc = {key: group_acc[key] for key in new_order_for_print}
    group_acc = {key: np.round(value, 4) for key, value in group_acc.items()}
    print(f"{print_label}:", str(group_acc))
    
    return losses.avg, acc.avg, group_acc    

def train_all_epochs(opt):
    best_acc = 0
    best_epoch = 0
    best_model = None
    # opt = parse_option()    
    
    if opt.tl_method in ["adapter_reg", "adapter_reg_seq", 'adapter_reg_seq_alter'] :
        print(f"====== TL:[{opt.tl_method}] LR:[{opt.learning_rate}] BS:[{opt.batch_size}] BSr:[{opt.batch_size_reg}] ======")
        reg_loader = deepcopy(opt.reg_loader)
    else:
        print(f"====== TL:[{opt.tl_method}] LR:[{opt.learning_rate}] BS:[{opt.batch_size}]======")
    
    print("> Simply copy the data loader ... ")
    # train_all_epochs 함수에서 다시 가져온다.
    # trainset = opt.trainset
    # train_loader = opt.train_loader
    # val_loader = opt.val_loader
    # test_loader = opt.test_loader
    
    # for Balancing Validation Loader.
    if opt.balance_val and (opt.tl_method in ["adapter_reg", "adapter_reg_seq", 'adapter_reg_seq_alter']):
        print("Using [Balanced] Validation loader for regularized training")
        origin_reg_loader = opt.reg_loader # -> From this loader, sampling balanced validation dataset for every epoch.
        
    if opt.resample_ce:
        print("Using [Resampled] Train loader for erm/feature laerning")
        opt.correct_class_bias = True
        opt.reweighting_by_class = False
        from visualizer_supcon import compute_slice_indices, prepare_contrastive_points, GetNegativesByClass, GetResampledWeightsCE
        from torch.utils.data.sampler import WeightedRandomSampler
        sliced_data_indices, sliced_data_correct = compute_slice_indices(opt, opt.trainset)
        contrastive_points = prepare_contrastive_points(opt.trainset,sliced_data_indices,sliced_data_correct)
        _, _, positives_by_class, _ = contrastive_points
    
        # skim_dataloader_by_group(train_loader)
        negatives_by_class = GetNegativesByClass(opt, opt.trainset, positives_by_class)
        weights_resampled_ce = GetResampledWeightsCE(opt.trainset, positives_by_class, negatives_by_class, opt)
        ce_sampler = WeightedRandomSampler(weights = weights_resampled_ce, num_samples = len(opt.trainset), replacement=True) # num_samples = len(trainset) -> oversampling 한 만큼 major group에서 unseen-sample 나옴
        resampled_train_loader = DataLoader(opt.trainset, sampler=ce_sampler, batch_size=opt.batch_size, num_workers=16)
        # skim_dataloader_by_group(resampled_train_loader)
    
    # group information
    get_yp_func = partial(get_y_p, n_places=opt.trainset.n_places)
    train_group_ratio = opt.trainset.group_ratio
    
    # build model and criterion

    classifier, criterion = set_model(opt) # model,  # CE
    # cl_loss = # Contrastive adpater

    # build optimizer
    print("Set Optimizer: SGD (default)")
    print('========================================================================')
    optimizer = set_optimizer(opt, classifier)
    
    # training routine
    train_losses = []
    train_accs = []
    train_group_accs = []
    
    val_losses = []
    val_accs = []
    val_group_accs = []
    
    test_losses = [] # NOTE: Don't peek ! 
    test_accs = [] # NOTE: Don't peek ! 
    test_group_accs = [] # NOTE: Don't peek ! 
    
    # entire training
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        print(f'--- Epoch {epoch} ---')
        
        # Sample Balanced Validation dataset.
        if opt.balance_val and (opt.tl_method in ["adapter_reg", "adapter_reg_seq", 'adapter_reg_seq_alter']):
            reg_loader = balance_val(origin_reg_loader, opt, print_procedure=False)
        
        # train one epoch
        if opt.tl_method == "adapter_reg":
            # Alternative Training
            if opt.use_cls_prompt_in_reg:
                loss, acc, group_acc = train_reg_one_epoch(opt, opt.train_loader, reg_loader, classifier, criterion, 
                                                            optimizer, epoch, get_yp_func, target=opt.train_target, group_prompt = False, print_label=f'Train (Alternative Learning)(Class prompt)')
            else:
                loss, acc, group_acc = train_reg_one_epoch(opt, opt.train_loader, reg_loader, classifier, criterion, 
                                                            optimizer, epoch, get_yp_func, target=opt.train_target, group_prompt = True, print_label=f'Train (Alternative Learning)(Group prompt)')
        elif opt.tl_method in  ["adapter_reg_seq", "adapter_reg_seq_alter"]:
            if epoch <= opt.epochs_feature_learning:
            # Sequetional Training
                loss, acc, group_acc = train_one_epoch(opt, opt.train_loader, classifier, criterion, 
                                                        optimizer, epoch, get_yp_func, target=opt.train_target, print_label=f'Train-1 (Feature Learning)')
                
            
            else:
                if epoch == opt.epochs_feature_learning + 1:
                    if opt.continue_from_best:
                        print("Load Best (Worst-acc) Model.")
                        classifier = deepcopy(best_model)
                    
                    if opt.add_adapter:
                        multiple_adapter, criterion = set_model_multiple_adapter(opt, classifier) # model,  # CE
                        optimizer_reg = set_optimizer_reg(opt, multiple_adapter) # Set new optimzer.
                        
                    else:
                        optimizer_reg = set_optimizer_reg(opt, classifier) # Set new optimzer.
                    
                adjust_learning_rate_reg(opt, optimizer_reg, epoch)
                
                if opt.tl_method == "adapter_reg_seq_alter":
                    if not opt.add_adapter:
                        if (epoch % 2) == 1:
                            loss, acc, group_acc = train_reg_seq_one_epoch(opt, reg_loader, classifier, criterion, 
                                                                    optimizer_reg, epoch, get_yp_func, target=opt.train_target, print_label=f'Train-2 (Balanced Learning)(Class prompt)', use_group=False)
                        else:
                            loss, acc, group_acc = train_reg_seq_one_epoch(opt, reg_loader, classifier, criterion, 
                                                                    optimizer_reg, epoch, get_yp_func, target=opt.train_target, print_label=f'Train-2 (Balanced Learning)(Group prompt)', use_group=True)
                    else:  
                        if (epoch % 2) == 1:
                            loss, acc, group_acc = train_reg_seq_one_epoch(opt, reg_loader, multiple_adapter, criterion, 
                                                                    optimizer_reg, epoch, get_yp_func, target=opt.train_target, print_label=f'Train-2 (Balanced Learning)(new adapter)(Class prompt)', use_group=False)
                        else:
                            loss, acc, group_acc = train_reg_seq_one_epoch(opt, reg_loader, multiple_adapter, criterion, 
                                                                    optimizer_reg, epoch, get_yp_func, target=opt.train_target, print_label=f'Train-2 (Balanced Learning)(new adapter)(Group prompt)', use_group=True)
                    
                else:
                    if not opt.add_adapter:
                        if not opt.use_cls_prompt_in_reg:
                            loss, acc, group_acc = train_reg_seq_one_epoch(opt, reg_loader, classifier, criterion, 
                                                                    optimizer_reg, epoch, get_yp_func, target=opt.train_target, print_label=f'Train-2 (Balanced Learning)(Group prompt)', use_group=True)
                        else:
                            loss, acc, group_acc = train_reg_seq_one_epoch(opt, reg_loader, classifier, criterion, 
                                                                    optimizer_reg, epoch, get_yp_func, target=opt.train_target, print_label=f'Train-2 (Balanced Learning)(Class prompt)', use_group=False)
                    else:
                        if not opt.use_cls_prompt_in_reg:
                            loss, acc, group_acc = train_reg_seq_one_epoch(opt, reg_loader, multiple_adapter, criterion, 
                                                                    optimizer_reg, epoch, get_yp_func, target=opt.train_target, print_label=f'Train-2 (Balanced Learning)(new adapter)(Group prompt)', use_group=True)
                        else:
                            loss, acc, group_acc = train_reg_seq_one_epoch(opt, reg_loader, multiple_adapter, criterion, 
                                                                    optimizer_reg, epoch, get_yp_func, target=opt.train_target, print_label=f'Train-2 (Balanced Learning)(new adapter)(Class prompt)', use_group=False)
                            
        else:
            loss, acc, group_acc = train_one_epoch(opt, opt.train_loader, classifier, criterion,
                          optimizer, epoch, get_yp_func, target=opt.train_target, print_label=f'Train({opt.train_target})')
        train_losses.append(loss); train_accs.append(acc); train_group_accs.append(group_acc)
        
        # eval for one epoch
        
        if opt.add_adapter and epoch > opt.epochs_feature_learning:
            val_loss, val_acc, val_group_acc = validate(opt, opt.val_loader, multiple_adapter, criterion, get_yp_func, train_group_ratio, target=opt.train_target, print_label=f'Val({opt.train_target})(new adapter)')
            val_losses.append(val_loss); val_accs.append(val_acc); val_group_accs.append(val_group_acc)
        else:
            val_loss, val_acc, val_group_acc = validate(opt, opt.val_loader, classifier, criterion, get_yp_func, train_group_ratio, target=opt.train_target, print_label=f'Val({opt.train_target})')
            val_losses.append(val_loss); val_accs.append(val_acc); val_group_accs.append(val_group_acc)
            
        # update best epoch by worst_group accuracy (default)
        if val_group_acc['worst_acc'] > best_acc:
            best_acc = val_group_acc['worst_acc']
            best_epoch = epoch
            
            if opt.add_adapter and epoch > opt.epochs_feature_learning:
                best_model = deepcopy(multiple_adapter)
            else:
                best_model = deepcopy(classifier)
                
        
        # test for one epoch
        if opt.add_adapter and epoch > opt.epochs_feature_learning:
            test_loss, test_acc, test_group_acc = validate(opt, opt.test_loader, multiple_adapter, criterion, get_yp_func, train_group_ratio, target='class', print_label=f'Test({opt.train_target})(new adapter)')
            test_losses.append(test_loss); test_accs.append(test_acc); test_group_accs.append(test_group_acc)
        else:
            test_loss, test_acc, test_group_acc = validate(opt, opt.test_loader, classifier, criterion, get_yp_func, train_group_ratio, target='class', print_label=f'Test({opt.train_target})')
            test_losses.append(test_loss); test_accs.append(test_acc); test_group_accs.append(test_group_acc)

    print('========================================================================')
    print("> end of training. \n")
    print('best epoch : {}'.format(best_epoch))
    
    best_train_group_acc = train_group_accs[best_epoch-1]
    best_val_group_acc = val_group_accs[best_epoch-1]
    best_test_group_acc = test_group_accs[best_epoch-1]
    
    print(f'best training accuracy on [{opt.train_target}]: {best_train_group_acc}')
    print(f'best validation accuracy on [{opt.train_target}]: {best_val_group_acc}')
    print(f'best test accuracy on [{opt.train_target}]: {best_test_group_acc}')
    
    # Evaluate Feature Quality using (Embedding-based) Zero-shot Prediction
    print('========================================================================')
    print("> start evaluating feature quality of best model. (using zero-shot prediction)\n")
    
    
    # Zero-shot [class] prediction
    zs_loss, zs_acc, zs_group_acc = validate_zs(opt, opt.test_loader, best_model, criterion, get_yp_func, train_group_ratio, target="class", print_label='zero-shot prediction (test) (class)')    
    
    if opt.tl_method in ["linear_probing"]:
        print(f" ㄴ Note that it should be same to [CLIP Zero-shot Baselines, of which worst acc is about 39%], in {opt.tl_method}")
    elif opt.tl_method in ["adapter", "contrastive_adapt"]: 
        print(f" ㄴ Note that it should be same to [best test accuracy on [{opt.train_target}]], above, in {opt.tl_method}")
    
    # Zero-shot [spurious] prediction 
    zs_loss_spurious, zs_acc_spurious, zs_group_acc_spurious = validate_zs(opt, opt.test_loader, best_model, criterion, get_yp_func, train_group_ratio, target="spurious", print_label='zero-shot prediction (test) (spurious)')    
    print(f" ㄴ Note that it is related to [richness of non-target (spurious) information] (-> 'mean_acc' is important)")
    
    print('========================================================================')
    # Recommendation : False when multiple training
    if opt.save_results and opt.num_iter == 1 :
        print('> Save results\n')
        all_results = {}
        
        for epoch in range(1, opt.epochs + 1):
            all_results[f"Epoch {epoch}"] = {"Train": train_group_accs[epoch-1], "Val": test_group_accs[epoch-1], "Test": test_group_accs[epoch-1]}
        
        final_results = {"Final Results (best epoch)":  {f"Epoch {best_epoch}": {"Train": best_train_group_acc, "Val": best_val_group_acc, "Test": best_test_group_acc}}, 
                         "Feature Quality (using zs)":  {"class":  zs_group_acc, "spurious": zs_group_acc_spurious}, 
                         "All Results (all epoch)": all_results}
        
        # make result folder 
        final_result_folder = os.path.dirname(opt.image_embedding_dir).replace('data', 'results')
        if not os.path.exists(final_result_folder):
            os.makedirs(final_result_folder)
            
        image_ebd_file_name = os.path.basename(opt.image_embedding_dir).split(".")[0]
        text_ebd_file_name = os.path.basename(opt.text_embedding_dir).split(".")[0]
        
        # result name
        final_result_file_name = f"im_{image_ebd_file_name}_t_{text_ebd_file_name}_tl_{opt.tl_method}_t_{opt.train_target}_lr_{opt.learning_rate}_bs_{opt.batch_size}"
        
        if "reg" in opt.tl_method:
            final_result_file_name += f"_lrr{opt.learning_rate_reg}_bsr_{opt.batch_size_reg}"

            if opt.balance_val:
                final_result_file_name += "_balval"
            
            
            if opt.tl_method != "adapter_reg_seq_alter": # Alter : GP <-> CP Alternative Training in Stage 2.
                if opt.use_cls_prompt_in_reg:
                    final_result_file_name+="_CP"
                else:
                    final_result_file_name+="_GP"
                
            if opt.add_adapter:
                final_result_file_name+="_MA"
                if not opt.init_near_identity:
                    final_result_file_name+="+rdm"
            
            if opt.continue_from_best and ('seq' in opt.tl_method):
                final_result_file_name+="_cont"
        
                 
        # NOTE This file name can be modified when we add baselines
        """
        E.g., if we use [flexable_adapter] and corresponding h.p. [flexable_weight], then, 
        if opt.tl_method == "flexable_adpater":
            final_result_file_name += f"_{opt.flexable_weight}"
        if opt.cosine:
            opt.model_name = '{}_cosine'.format(opt.model_name)
        if opt.warm:
            opt.model_name = '{}_warm'.format(opt.model_name)
        """

        # result path
        final_result_file_path = os.path.join(final_result_folder, final_result_file_name + ".json")
        final_model_path = os.path.join(final_result_folder, final_result_file_name + ".pth")
        
        print('final result path: ', final_result_file_path)
        print('final model path: ', final_model_path)
        
        # save results, as json.
        with open(final_result_file_path, "w") as f:
            json.dump(final_results, f, indent=4)
        
        # save final model, as pth 
        torch.save(best_model.state_dict(), final_model_path)    
            
    
    print('========================================================================')
    print("> end")
    # del trainset; del train_loader; del val_loader; del test_loader
    # if opt.tl_method in ["adapter_reg", "adapter_reg_seq", 'adapter_reg_seq_alter'] :
    #     del reg_loader
    
    return (best_train_group_acc, best_val_group_acc, best_test_group_acc), (zs_group_acc, zs_group_acc_spurious)

if __name__ == '__main__' :
    opt = parse_option()
    
    # Data Loader: Model 당 한 번만 수행하자. (CPU 아파 죽는 것 같음. num_workers 16 / 
    
    print(f"> Start Transfer Learning using [{opt.tl_method}]")
    print('========================================================================')
    if opt.dataset == 'waterbirds':
        # build data loader                
        if opt.tl_method in ["adapter_reg", "adapter_reg_seq", 'adapter_reg_seq_alter'] :
            from data.waterbirds_embeddings_reg import WaterbirdsEmbeddings, load_waterbirds_embeddings
            print(f"Load image embedding of Waterbirds: {opt.image_embedding_dir}")
            opt.trainset = WaterbirdsEmbeddings(opt.data_dir, 'train', opt.image_embedding_dir, None)
            print("Load Data Loader (train, validation, test)")
            opt.train_loader, opt.reg_loader, opt.val_loader, opt.test_loader = load_waterbirds_embeddings(opt.data_dir, opt.image_embedding_dir, opt.batch_size, opt.batch_size_reg)
        else:
            from data.waterbirds_embeddings import WaterbirdsEmbeddings, load_waterbirds_embeddings
            print(f"Load image embedding of Waterbirds: {opt.image_embedding_dir}")
            trainset = WaterbirdsEmbeddings(opt.data_dir, 'train', opt.image_embedding_dir, None)
            print(f"ㄴ Corresponding text embedding of Waterbirds: {opt.text_embedding_dir}")
            opt.train_loader, opt.val_loader, opt.test_loader = load_waterbirds_embeddings(opt.data_dir, opt.image_embedding_dir, opt.batch_size, opt.batch_size)
        
        if opt.train_target == "class":
            print(f"Training target : {opt.train_target} (Land bird(0) / Water bird(1))")
        elif opt.train_target == "spurious":
            print(f"Training target : {opt.train_target} (Land background(0) / Water background(1))")
        
    elif opt.dataset == 'celeba':
        # build dataset example.
        from data.celeba_embeddings import CelebaEmbeddings, load_celeba_embeddings # 버근가.. 왜 인식을 몬하지
        print(f"Load embedding of CelebA: {opt.image_embedding_dir}")
        
        opt.trainset = CelebaEmbeddings(opt.data_dir, 'train', opt.image_embedding_dir, None)
        
        # build data loader
        print("Load Data Loader (train, validation, test)")
        
        train_loader, val_loader, test_loader = load_celeba_embeddings(opt.data_dir, opt.image_embedding_dir, opt.batch_size, opt.batch_size)
        if opt.tl_method in ["adapter_reg", "adapter_reg_seq", 'adapter_reg_seq_alter'] :
            from data.celeba_embeddings_reg import CelebaEmbeddings, load_celeba_embeddings
            print(f"Load embedding of CelebA: {opt.image_embedding_dir}")
            opt.trainset = CelebaEmbeddings(opt.data_dir, 'train', opt.image_embedding_dir, None)
            print("Load Data Loader (train, validation, test)")
            opt.train_loader, opt.reg_loader, opt.val_loader, opt.test_loader = load_celeba_embeddings(opt.data_dir, opt.image_embedding_dir, opt.batch_size, opt.batch_size_reg)
        else:
            print(f"Load image embedding of Waterbirds: {opt.image_embedding_dir}")
            opt.trainset = CelebaEmbeddings(opt.data_dir, 'train', opt.image_embedding_dir, None)
            print(f"ㄴ Corresponding text embedding of Waterbirds: {opt.text_embedding_dir}")
            opt.train_loader, opt.val_loader, opt.test_loader = load_celeba_embeddings(opt.data_dir, opt.image_embedding_dir, opt.batch_size, opt.batch_size)
        
        # print training target
        if opt.train_target == "class":
            print(f"Training target : {opt.train_target} (non-blond hair(0) / blond hair(1))")
        elif opt.train_target == "spurious":
            print(f"Training target : {opt.train_target} (female(0) / male(1))")

    # # train_all_epochs 함수에서 다시 가져온다.
    # opt.trainset = trainset
    # opt.train_loader = train_loader
    # opt.val_loader = val_loader
    # opt.test_loader = test_loader
    
    # if opt.tl_method in ["adapter_reg", "adapter_reg_seq", 'adapter_reg_seq_alter'] :
    #     opt.reg_loader = reg_loader
    
    
    # set the path according to the environment
    lr_list = opt.lr_list.split(',')
    lr_list = [float(lr) for lr in lr_list]
    bs_list = opt.bs_list.split(',')
    bs_list = [int(bs) for bs in bs_list]
    
    # for adapter_reg, adapter_reg_seq, adapter_reg_seq_ma.
    bsr_list = opt.bsr_list.split(',')
    bsr_list = [int(bsr) for bsr in bsr_list]
    
    if opt.tl_method =="adapter":
        bsr_list = [128] # 혹시 argument 잘못 넣었을 때를 대비.
        
    for lr in lr_list:
        for bs in bs_list:
            for bsr in bsr_list:
                opt.learning_rate = lr
                opt.learning_rate_reg = lr * opt.lr_multiple
                opt.batch_size = bs
                opt.batch_size_reg = bsr
    
                for iter in range(1, opt.num_iter+1):     
                    print(f"f=============Iteration : {iter}/{opt.num_iter}=============")        
                    
                    set_seed(opt.random_seeds[iter-1])
                    
                    (tr_group_acc, val_group_acc, test_group_acc), (zs_target, zs_spurious)= train_all_epochs(opt)
                    
                    if iter==1:
                        tr_df = pd.DataFrame(tr_group_acc, index=[iter])
                        val_df = pd.DataFrame(val_group_acc, index=[iter])
                        test_df = pd.DataFrame(test_group_acc, index=[iter])
                        zs_target_df = pd.DataFrame(zs_target, index=[iter])
                        zs_spurious_df = pd.DataFrame(zs_spurious, index=[iter])
                    else:
                        tr_df = pd.concat([tr_df, pd.DataFrame(tr_group_acc, index=[iter])])
                        val_df = pd.concat([val_df, pd.DataFrame(val_group_acc, index=[iter])])
                        test_df = pd.concat([test_df, pd.DataFrame(test_group_acc, index=[iter])])
                        zs_target_df = pd.concat([zs_target_df, pd.DataFrame(zs_target, index=[iter])])
                        zs_spurious_df = pd.concat([zs_spurious_df, pd.DataFrame(zs_spurious, index=[iter])])
                    
                tr_df = pd.concat([tr_df, pd.DataFrame(tr_df.mean().to_dict(), index=['tr_mean'])])
                tr_df = pd.concat([tr_df, pd.DataFrame(tr_df.std().to_dict(), index=['tr_std'])])
                val_df = pd.concat([val_df, pd.DataFrame(val_df.mean().to_dict(), index=['val_mean'])])
                val_df = pd.concat([val_df, pd.DataFrame(val_df.std().to_dict(), index=['val_std'])])
                test_df = pd.concat([test_df, pd.DataFrame(test_df.mean().to_dict(), index=['test_mean'])])
                test_df = pd.concat([test_df, pd.DataFrame(test_df.std().to_dict(), index=['test_std'])])
                zs_target_df = pd.concat([zs_target_df, pd.DataFrame(zs_target_df.mean().to_dict(), index=['zs_tg_mean'])])
                zs_target_df = pd.concat([zs_target_df, pd.DataFrame(zs_target_df.std().to_dict(), index=['zs_tg_std'])])
                zs_spurious_df = pd.concat([zs_spurious_df, pd.DataFrame(zs_spurious_df.mean().to_dict(), index=['zs_spu_mean'])])
                zs_spurious_df = pd.concat([zs_spurious_df, pd.DataFrame(zs_spurious_df.std().to_dict(), index=['zs_spu_std'])])
                
                
                final_df = pd.concat([test_df, zs_spurious_df, tr_df, val_df, zs_target_df])
                result_root = "results_iterative"
                if not os.path.exists(result_root):
                    os.mkdir(result_root)
                
                final_result_file_path = f"ds_{opt.dataset}_tl_{opt.tl_method}_bs_{opt.batch_size}_lr_{opt.learning_rate}"
                
                if "reg" in opt.tl_method:
                    final_result_file_path += f"_lrr{opt.learning_rate_reg}_bsr{opt.batch_size_reg}"
                
                    if opt.balance_val:
                        final_result_file_path += "_balval"
                    
                    if opt.tl_method != "adapter_reg_seq_alter":      
                        if opt.use_cls_prompt_in_reg:
                            final_result_file_path += f"_CP"
                        else:
                            final_result_file_path += f"_GP"
                        
                    if opt.add_adapter:
                        final_result_file_path += f"_MA"
                        if opt.init_near_identity:
                            final_result_file_path += "+ni"
                        else:
                            final_result_file_path += "+rn"
                    
                    if opt.continue_from_best and ('seq' in opt.tl_method):
                        final_result_file_name+="_cont"
                
                
                if opt.resample_ce:
                    final_result_file_path+="_rs"
                
                final_df = final_df.round(4)
                print("Final Results: ", final_df)
                print("Save to: ", os.path.join(result_root, final_result_file_path)+'.csv')
                final_df.to_csv(os.path.join(result_root, final_result_file_path)+'.csv')
                