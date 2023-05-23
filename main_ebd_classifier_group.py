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

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer

from data.waterbirds_embeddings import WaterbirdsEmbeddings, load_waterbirds_embeddings
from data.celeba_embeddings import CelebaEmbeddings, load_celeba_embeddings
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
        
        self.text_features = get_text_embedding(self.text_embedding_dir)
        self.n_cls = self.text_features.shape[0]
        self.text_spurious_features = get_text_embedding(self.text_spurious_embedding_dir)
        
    def forward(self, features, use_group=False): 
        image_features =  self.adapter(features) # Un-normalized (B, 1024)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # Normalized (B, 1024)

        #NOTE Joonwon Added
        if use_group:
            text_features = get_text_embedding(self.text_group_embedding_dir) # (Pre) Normalized (B, 2, 1024)
        else:
            text_features = self.text_features # (Pre) Normalized (B, 2, 1024)
        
        # Check if we have to normalize the text features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = image_features @ text_features / self.temperature # (B, 1024) X (B, C, 1024) = # (B, C)
        
        return logits
    
    def forward_spurious(self, features): 
        image_features =  self.adapter(features) # Un-normalized (B, 1024)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # Normalized (B, 1024)

        text_spurious_features = self.text_spurious_features # (Pre) Normalized (B, 2, 1024)
        
        logits = image_features @ text_spurious_features / self.temperature # (B, 1024) X (B, 2, 1024) = # (B, 2)
        
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
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3, 
                        help='learning rate') # Tuning needed. 
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

    parser.add_argument('--image_embedding_dir', type=str,
                        help='extracted image embedding')
    parser.add_argument('--text_embedding_dir', type=str,
                        help='extracted text embedding')
    parser.add_argument('--text_spurious_embedding_dir', type=str,
                        help='extracted text embedding (about spurious attributes)')
    parser.add_argument('--train_target', type=str, default="class", choices=["class", "spurious", "group"]) # label for prediction.
    parser.add_argument('--data_dir', type=str,
                    help='folder, in which metadata.csv] exist')
    parser.add_argument('--tl_method', type=str, default= "linear_probing", choices=["linear_probing", "adapter", "contrastive_adapter", "ETC"]
                        ,help='transfer learning method')
    parser.add_argument('--adapter_feat_dim', type=int, default= 128, help='reduced dimension in adapter')
    parser.add_argument('--zs_temperature', type=float, default= 0.01, help='Temperature in zero-shot prediction')
    parser.add_argument('--watch_batch_results', type=bool, default=False, help='Print results in each bach by [opt.print_freq]. Recommdned: True when single-run of CelebA(Large # of batch), False others')
    parser.add_argument('--save_results', type=bool, default=True, help='Save the results of transfer learning (and final feature quality) in the folder where ')
    

    # parser.add_argument('--lr_linear_probing', type=float, default=1e-3, chocies=[1e-3, 1e-2, 1e-1, 1, 3, 10], help='learning rate for linear probing') # Tuning needed. 
      # -> Zero-shot으로 대체하는 게 맞을듯.

    opt = parser.parse_args()
    
    if opt.

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
            
    if opt.dataset == 'celeba':
        opt.n_cls = 2
    elif opt.dataset == 'waterbirds':
        opt.n_cls = 2
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

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
        classifier = CustomCLIP(adapter, opt.text_embedding_dir, opt.text_spurious_embedding_dir, temperature=opt.zs_temperature)

    if torch.cuda.is_available():
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return classifier, criterion # model, 

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
    text_features = torch.stack(text_features, dim=1).cuda() # (B, 2, 1024)
    
    
    return text_features

#NOTE 이거랑 유사한 함수를 하나 더 만들거나, if 문 넣어서 if predict_group: criterion(embeddings, group) else: criterion(embeddings, labels)
def train_one_epoch(opt, train_loader, classifier, criterion, optimizer, epoch, get_yp_func, target, print_label='Train', predict_group = True): # model,
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
        
        if opt.watch_batch_results:
            if (idx + 1) % opt.print_freq == 0:
                print(f'{print_label}: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'Acc@1 {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, acc=acc))
                sys.stdout.flush()
            
    group_acc = get_results(acc_groups, get_yp_func) # NOTE declared in [def main]
    group_acc = {key: group_acc[key] for key in new_order_for_print[1:]}
    group_acc = {key: np.round(value, 4) for key, value in group_acc.items()}
    print(f"{print_label}:", str(group_acc))
    
    return losses.avg, acc.avg, group_acc

# NOTE joonwon added
def train_reg_one_epoch(opt, train_loader1, train_loader2, classifier, criterion, optimizer, epoch, get_yp_func, target, print_label='Train', predict_group = True): # model,
    """one epoch training with regulalizar"""
    # model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_groups = {g_idx : AverageMeter() for g_idx in range(train_loader1.dataset.n_groups)}

    end = time.time()


    for dataloader, use_group in zip([train_loader1, train_loader2], [False, True]):
        for idx, data in enumerate(dataloader):  
            
            embeddings, all_labels, img_filenames = data # all_labels.keys() : ['class', 'group', 'spurious', 'ebd_pred'(CLIP-zeroshot)] 
            labels = all_labels[target] # target : one of [y, spurious, group]
            groups = all_labels['group'] # For evaluating group accuracy (and further developing group-information-aware approaches)
        
            data_time.update(time.time() - end)

            embeddings = embeddings.cuda(non_blocking=True)
            # NOTE joonwon added
            if use_group:
                labels = groups
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # warm-up learning rate
            warmup_learning_rate(opt, epoch, idx, len(dataloader), optimizer)

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

def validate(opt, val_loader, classifier, criterion, get_yp_func, train_group_ratio, target, print_label='Test'):
    """validation"""
    
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
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
        
            if opt.watch_batch_results:
                if (idx+1) % opt.print_freq == 0:
                    print(f'{print_label}: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {acc.val:.3f} ({acc.avg:.3f})'.format(
                        idx, len(val_loader), batch_time=batch_time,
                        loss=losses, acc=acc))
                    
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



    

def validate_zs(opt, val_loader, classifier, criterion, get_yp_func, train_group_ratio, target, print_label='Zero-shot Prediction (Test) (Class)'):
    """(Feature quality) validation using zeroshot-prediction"""

    classifier.eval()


    if opt.tl_method in ["linear_probing"]:
        temperature = opt.zs_temperature
        
        if target=="class":
            text_embeddings = get_text_embedding(opt.text_embedding_dir)
        elif target=='spurious':
            text_embeddings = get_text_embedding(opt.text_spurious_embedding_dir)
        
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
                
            elif opt.tl_method in ['adapter', 'contrastive_adapter']: # Adpater, Contrastive Adapter : Embedding -> (1) (Adapted) Embedding -> (2) ZeroShot prediction as logit    (CustomCLIP.forward : (1)+(2))
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
        
            if opt.watch_batch_results:
                if (idx+1) % opt.print_freq == 0:
                    print(f'{print_label}: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {acc.val:.3f} ({acc.avg:.3f})'.format(
                        idx, len(val_loader), batch_time=batch_time,
                        loss=losses, acc=acc))
                    
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
    
    
    print(f"> Start Transfer Learning using [{opt.tl_method}]")
    print('========================================================================')
    if opt.dataset == 'waterbirds':
        # build dataset example.
        print(f"Load image embedding of Waterbirds: {opt.image_embedding_dir}")
        trainset = WaterbirdsEmbeddings(opt.data_dir, 'train', opt.image_embedding_dir, None)
        print(f"ㄴ Corresponding text embedding of Waterbirds: {opt.text_embedding_dir}")
        # build data loader
        print("Load Data Loader (train, validation, test)")
        if opt.tl_method == "adapter_reg":
            from data.waterbirds_embeddings_reg import WaterbirdsEmbeddings, load_waterbirds_embeddings
            train_loader, reg_loader, val_loader, test_loader = load_waterbirds_embeddings(opt.data_dir, opt.image_embedding_dir, opt.batch_size, opt.batch_size)
        else:
            train_loader, val_loader, test_loader = load_waterbirds_embeddings(opt.data_dir, opt.image_embedding_dir, opt.batch_size, opt.batch_size)
        if opt.train_target == "class":
            print(f"Training target : {opt.train_target} (Land bird(0) / Water bird(1))")
        elif opt.train_target == "spurious":
            print(f"Training target : {opt.train_target} (Land background(0) / Water background(1))")
        
    elif opt.dataset == 'celeba':
        # build dataset example.
        print(f"Load embedding of CelebA: {opt.image_embedding_dir}")
        trainset = CelebaEmbeddings(opt.data_dir, 'train', opt.image_embedding_dir, None)
        
        # build data loader
        print("Load Data Loader (train, validation, test)")
        train_loader, val_loader, test_loader = load_celeba_embeddings(opt.data_dir, opt.image_embedding_dir, opt.batch_size, opt.batch_size)
        
        # print training target
        if opt.train_target == "class":
            print(f"Training target : {opt.train_target} (non-blond hair(0) / blond hair(1))")
        elif opt.train_target == "spurious":
            print(f"Training target : {opt.train_target} (female(0) / male(1))")

    # group information
    get_yp_func = partial(get_y_p, n_places=trainset.n_places)
    train_group_ratio = trainset.group_ratio
    
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
        
        # train one epoch
        if opt.tl_method == "adapter_reg":
            loss, acc. group_acc = train_reg_one_epoch(opt, train_loader, reg_loader, classifier, criterion,
                          optimizer, epoch, get_yp_func, target=opt.train_target, print_label=f'Train({opt.train_target})')
        else:
            loss, acc, group_acc = train_one_epoch(opt, train_loader, classifier, criterion,
                          optimizer, epoch, get_yp_func, target=opt.train_target, print_label=f'Train({opt.train_target})')
        train_losses.append(loss); train_accs.append(acc); train_group_accs.append(group_acc)
        
        # eval for one epoch
        val_loss, val_acc, val_group_acc = validate(opt, val_loader, classifier, criterion, get_yp_func, train_group_ratio, target=opt.train_target, print_label=f'Val({opt.train_target})')
        val_losses.append(val_loss); val_accs.append(val_acc); val_group_accs.append(val_group_acc)
        
        # update best epoch by worst_group accuracy (default)
        if val_group_acc['worst_acc'] > best_acc:
            best_acc = val_group_acc['worst_acc']
            best_epoch = epoch
            best_model = deepcopy(classifier)
        
        # test for one epoch
        test_loss, test_acc, test_group_acc = validate(opt, test_loader, classifier, criterion, get_yp_func, train_group_ratio, target='class', print_label=f'Test({opt.train_target})')
        
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
    zs_loss, zs_acc, zs_group_acc = validate_zs(opt, test_loader, best_model, criterion, get_yp_func, train_group_ratio, target="class", print_label='zero-shot prediction (test) (class)')    
    
    if opt.tl_method in ["linear_probing"]:
        print(f" ㄴ Note that it should be same to [CLIP Zero-shot Baselines, of which worst acc is about 39%], in {opt.tl_method}")
    elif opt.tl_method in ["adapter", "contrastive_adapt"]: 
        print(f" ㄴ Note that it should be same to [best test accuracy on [{opt.train_target}]], above, in {opt.tl_method}")
    
    # Zero-shot [spurious] prediction 
    zs_loss_spurious, zs_acc_spurious, zs_group_acc_spurious = validate_zs(opt, test_loader, best_model, criterion, get_yp_func, train_group_ratio, target="spurious", print_label='zero-shot prediction (test) (spurious)')    
    print(f" ㄴ Note that it is related to [richness of non-target (spurious) information] (-> 'mean_acc' is important)")
    
    print('========================================================================')
    # Recommendation : False when multiple training
    if opt.save_results:
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
    
    return (best_train_group_acc, best_val_group_acc, best_test_group_acc), (train_group_accs, val_group_accs, test_group_accs) # (best_results, all_results)

if __name__ == '__main__' :
    opt = parse_option()
    train_all_epochs(opt)