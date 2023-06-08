from __future__ import print_function

from visualizer import *
import numpy as np
import pandas as pd
import os
import sys
import json
import torch
import torch.nn as nn

import sys
import argparse
import time
from tqdm import tqdm
import math
import copy

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

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

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, features):
        return self.fc(features)



# class CustomCLIP(nn.Module):
#     def __init__(self, adapter, text_embedding_dir, text_spurious_embedding_dir, temperature=0.01, head_method = None, ca_feat_dim = 128, ca_pre_norm = True):
#         super().__init__()
#         self.text_embedding_dir = text_embedding_dir
#         self.text_spurious_embedding_dir = text_spurious_embedding_dir
#         self.adapter = adapter
#         self.ca_feat_dim = ca_feat_dim
#         self.ca_pre_norm = ca_pre_norm
#         # Contrastive Adapter
        
#         self.head_method = head_method
        
#         if self.head_method == 'linear':
#             self.head = nn.Linear(self.adapter.input_dim, ca_feat_dim)

#         self.temperature = temperature # CA default : 0.01, B2T default : 0.02 (?) NOTE
        
#         self.text_features = get_text_embedding(self.text_embedding_dir).cuda() # (1024, 2)
#         self.text_spurious_features = get_text_embedding(self.text_spurious_embedding_dir).cuda() # (1024, 2)
        
#     def forward(self, features): 
#         image_features =  self.adapter(features) # Un-normalized (B, 1024)
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True) # Normalized (B, 1024)

#         text_features = self.text_features / self.text_features.norm(dim=0, keepdim=True) # Normalized # (1024, 2)
        
#         logits = image_features @ text_features / self.temperature # (B, 1024) X (1024, 2) = # (B, 2)
        
#         return logits
    
#     def forward_spurious(self, features): 
#         image_features =  self.adapter(features) # Un-normalized (B, 1024)
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True) # Normalized (B, 1024)

#         text_spurious_features = self.text_spurious_features / self.text_spurious_features.norm(dim=0, keepdim=True) #Normalized (1024, 2)
        
#         logits = image_features @ text_spurious_features / self.temperature # (B, 1024) X (1024, 2) = # (B, 2)
        
#         return logits
    
#     def forward_ca(self, x):
#         if self.ca_pre_norm:
#             x = x / x.norm(dim=-1, keepdim=True) # 일단 Encoder 이전에 Feature norm도 해주어야함. (Default..)
            
#         feat = self.adapter(x)
#         if self.head_method == 'linear':
#             proj = self.head(feat)
#         else:
#             proj = feat
            
#         proj = proj / proj.norm(dim=-1, keepdim=True)
#         return proj
        
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
class Adapter(nn.Module):
    """
    - Residual connetion : 제외 (original Adapter - 0.2*images + 0.8*adapter)
    - Hidden dimension : args.adapter_feat_dim (original Adatper - input_dim // 4)
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
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
    parser.add_argument('--ca_head', type=str, default= None, help='projection head for contrastive loss')
    parser.add_argument('--contrastive_weight', type=float, default=0.5, help='balancing weight between Sup-con <-> supervised CE')
    parser.add_argument('--zs_temperature', type=float, default= 0.01, help='Temperature in zero-shot prediction')
    parser.add_argument('--watch_batch_results', type=bool, default=False, help='Print results in each bach by [opt.print_freq]. Recommdned: True when single-run of CelebA(Large # of batch), False others')
    parser.add_argument('--save_results', type=bool, default=True, help='Save the results of transfer learning (and final feature quality) in the folder where ')
    

    # parser.add_argument('--lr_linear_probing', type=float, default=1e-3, chocies=[1e-3, 1e-2, 1e-1, 1, 3, 10], help='learning rate for linear probing') # Tuning needed. 
      # -> Zero-shot으로 대체하는 게 맞을듯.

    opt = parser.parse_args(args=[])

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


    opt.cl_temperature = 0.1
    # opt.ca_head = "linear" # linear or mlp    
    opt.ca_feat_dim = 128 
    
    opt.print_freq_ca = 1
    opt.contrastive_weight = 0.1
    
    
    opt.ca_pre_norm = True
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

def get_text_embedding(text_embedding_dir, return_key = False):
    with open(text_embedding_dir, 'r') as f:
        text_embeddings = json.load(f)

    text_features = []
    class_templates = []
    for class_template, class_embedding in text_embeddings.items():
        text_features.append(torch.tensor(class_embedding))
        class_templates.append(class_template)
    
    if not return_key:
        text_features = torch.stack(text_features, dim=1) # (1024, 2)
        return text_features
    else:
        list_template_feature_pair = []
        for idx in range(len(text_features)):
            list_template_feature_pair.append({class_templates[idx]: text_features[idx].numpy()})
        return list_template_feature_pair
        
        
    

def train_one_epoch(opt, train_loader, classifier, criterion, optimizer, epoch, get_yp_func, target, print_label='Train'): # model,
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
        if idx>=opt.ce_update:
            continue
        embeddings, all_labels, img_filenames = data # all_labels.keys() : ['class', 'group', 'spurious', 'ebd_pred'(CLIP-zeroshot)] 
        labels = all_labels[target] # target : one of [y, spurious, group]
        groups = all_labels['group'] # For evaluating group accuracy (and further developing group-information-aware approaches)
    
        data_time.update(time.time() - end)

        embeddings = embeddings.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        
        # NOTE Embedding 추가
        # embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = classifier(embeddings.detach()) 
        loss = criterion(output, labels)
        # balancing weight
        
        # loss = (1-opt.contrastive_weight) * loss
        # update metric
        losses.update(loss.item(), bsz)
        acc1 = accuracy(output, labels, bsz)
        acc.update(acc1, bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # NOTE
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

def train_one_epoch_cl(opt, train_loader, classifier, contrastive_loss, optimizer, epoch, print_label='Train'):
    """
    Train contrastive epoch
    """
    classifier.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    poss = AverageMeter()
    negs = AverageMeter()
    # acc = AverageMeter()
    # acc_groups = {g_idx : AverageMeter() for g_idx in range(train_loader.dataset.n_groups)}

    # contrastive_weight = args.contrastive_weight
    loss_compute_size = int(opt.num_anchor +
                            opt.num_negative +
                            opt.num_positive)
    
    end = time.time()
    # 330k -> (1+819+819) * opt.batch_factor ~ 52k
    for idx, batch_data in enumerate(train_loader):
        if idx>=opt.ca_update:
            continue
        # Setup main contrastive batch
        ## 순서대로 임베딩, 
        all_batch_inputs, all_batch_labels, _ = batch_data
        
        all_batch_inputs = all_batch_inputs.cuda(non_blocking=True) 
        # 각각 1개의 Anchor, N개의 Positive , M개의 Negative
        # 총 opt.batch_factor 개의 Triplet. 
        batch_inputs = torch.split(all_batch_inputs,
                                  loss_compute_size)
        
        # 기본적인 CA에선 사용 x
        # all_batch_labels, all_batch_group, all_batch_spurious, all_batch_zs_pred = all_batch_labels.values()

        data_time.update(time.time() - end)

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        
        neg_start_ix = opt.num_anchor + opt.num_positive
        neg_end_ix = neg_start_ix + opt.num_negative
        
        optimizer.zero_grad()
        for ix, batch_input in enumerate(batch_inputs):
            inputs_a = batch_input[:opt.num_anchor]
            inputs_p = batch_input[opt.num_anchor:neg_start_ix]
            inputs_n = batch_input[neg_start_ix:neg_end_ix]

            # Just do contrastive loss against first anchor for now
            inputs_a_ = [inputs_a[0]] # [1, 1024]

            # in Contrastive Adapter, iterated over only "single" anchor
            for anchor_ix, input_a in enumerate(inputs_a_):
                contrastive_batch = torch.vstack((input_a.unsqueeze(0),
                                                  inputs_p, inputs_n))
                # compute loss
                # loss = contrastive_loss(classifier, contrastive_batch) # anchor 1개에 대한 Loss
                loss, pos_numerator_last, neg_numerator_last, denominator_last = contrastive_loss(classifier, contrastive_batch) # anchor 1개에 대한 Loss # 관찰용
                contrastive_batch = contrastive_batch.detach().cpu()
                
            # update metric
            # loss *= (1/(len(inputs_a) * len(batch_inputs))) #NOTE Scaling 일단 빼는 게 맞을듯. (32로 나누는 거) -> 딱히 배치로스를 안 씀. 그리고 이거 강제로 Loss 낮춰서 supervision 약해지지 않나? 
            loss = (opt.contrastive_weight) * loss # NOTE Balancing.
            
            loss = loss / (opt.batch_factor)
            losses.update(loss.item(), 1)
            poss.update(pos_numerator_last.item(), 1)
            negs.update(neg_numerator_last.item(), 1)

            # SGD      
            loss.backward()
            

            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Update acc dict
            # update_dict(acc_groups, labels, groups, output)
        
        optimizer.step()
        optimizer.zero_grad()
        if opt.watch_batch_results:
            if (idx + 1) % (opt.print_freq_ca) == 0:
                print(f'{print_label}: [{epoch}][{idx+1}/{len(train_loader)}][{ix+1}/{len(batch_inputs)}]\t'
                    f'scaled_exp_pos {pos_numerator_last:.3f}\t'
                    f'scaled_exp_neg {neg_numerator_last:.3f}\t'
                    f'loss {losses.val:.3f} ({losses.avg:.3f})\t')
       
    print(f"Loss in {print_label}:", f"{losses.avg:.3f}", f"Pos:{poss.avg:.4f}, Neg: {negs.avg:.4f}")


    return losses.avg # , acc.avg, group_acc


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
            
            # NOTE Embedding 추가
            # embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            # forward
            output = classifier(embeddings)
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
        text_embeddings = text_embeddings / text_embeddings.norm(dim=0, keepdim=True) # (1024, 2)

        
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
                output = image_embeddings @ text_embeddings / temperature # (B, 1024) X (1024, 2) = # (B, 2)
                
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
    
    
    print(f"> Start Transfer Learning using [{opt.tl_method}]")
    print('========================================================================')
    if opt.dataset == 'waterbirds':
        # build dataset example.
        print(f"Load image embedding of Waterbirds: {opt.image_embedding_dir}")
        trainset = WaterbirdsEmbeddings(opt.data_dir, 'train', opt.image_embedding_dir, None)
        print(f"ㄴ Corresponding text embedding of Waterbirds: {opt.text_embedding_dir}")
        # build data loader
        print("Load Data Loader (train, validation, test)")
        train_loader, val_loader, test_loader = load_waterbirds_embeddings(opt.data_dir, opt.image_embedding_dir, opt.batch_size, opt.batch_size)
        
        # print training target
        if opt.train_target == "class":
            print(f"Training target : {opt.train_target} (Land bird(0) / Water bird(1))")
        elif opt.train_target == "spurious":
            print(f"Training target : {opt.train_target} (Land background(0) / Water background(1))")
        
    elif opt.dataset == 'celeba':
        # build dataset example.
        print(f"Load embedding of CelebA: {opt.image_embedding_dir}")
        trainset = CelebaEmbeddings(opt.data_dir, 'train', opt.image_embedding_dir, None)
        print(f"ㄴ Corresponding text embedding of Waterbirds: {opt.text_embedding_dir}")
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
    classifier, criterion = set_model(opt) # model, 

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
            best_model = copy.deepcopy(classifier)
        
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


def train_epoch(encoder, classifier, dataloader,
                optim_e, optim_c, scheduler_e, scheduler_c,
                epoch, val_loader, contrastive_loss,
                cross_entropy_loss, args):
    """
    Train contrastive epoch
    """
    encoder.to(args.device)
    classifier.to(args.device)

    optim_e.zero_grad()
    optim_c.zero_grad()
    contrastive_weight = args.contrastive_weight
    loss_compute_size = int(args.num_anchor +
                            args.num_negative +
                            args.num_positive +
                            args.num_negative_easy)
    epoch_losses = []
    epoch_losses_contrastive = []
    epoch_losses_cross_entropy = []

    encoder.eval()
    classifier.train()

    total_updates = int(len(dataloader) * args.batch_factor)
    pbar = tqdm(total=total_updates)
    for batch_ix, batch_data in enumerate(dataloader):

        batch_loss = 0
        batch_loss_contrastive = 0
        batch_loss_cross_entropy = 0
        batch_loss_kl = 0
        batch_count = 0

        # Setup main contrastive batch
        all_batch_inputs, all_batch_labels, all_batch_indices = batch_data
        batch_inputs = torch.split(all_batch_inputs,
                                   loss_compute_size)
        batch_labels = torch.split(all_batch_labels,
                                   loss_compute_size)
        batch_indices = np.split(all_batch_indices, len(batch_inputs))

        if args.supervised_linear_scale_up:
            supervised_weight = ((1 - args.contrastive_weight) *
                                 ((epoch * len(dataloader) + batch_ix) *
                                 args.supervised_step_size))
        elif epoch < args.supervised_update_delay:
            supervised_weight = 0
        else:
            supervised_weight = 1 - args.contrastive_weight

        for ix, batch_input in enumerate(batch_inputs):
            neg_start_ix = args.num_anchor + args.num_positive
            neg_end_ix = neg_start_ix + args.num_negative

            inputs_a = batch_input[:args.num_anchor]
            inputs_p = batch_input[args.num_anchor:neg_start_ix]
            inputs_n = batch_input[neg_start_ix:neg_end_ix]
            inputs_ne = batch_input[-args.num_negative_easy:]

            labels_a = batch_labels[ix][:args.num_anchor]
            labels_p = batch_labels[ix][args.num_anchor:neg_start_ix]
            labels_n = batch_labels[ix][neg_start_ix:neg_end_ix]
            labels_ne = batch_labels[ix][-args.num_negative_easy:]

            # Just do contrastive loss against first anchor for now
            inputs_a_ = [inputs_a[0]]
            for anchor_ix, input_a in enumerate(inputs_a_):
                contrastive_batch = torch.vstack((input_a.unsqueeze(0),
                                                  inputs_p, inputs_n))
                # Compute contrastive loss
                loss = contrastive_loss(encoder, contrastive_batch)
                loss *= ((1 - supervised_weight) /
                         (len(inputs_a_) * len(batch_inputs)))
                loss.backward()
                contrastive_batch = contrastive_batch.detach().cpu()

                batch_loss += loss.item()
                batch_loss_contrastive += loss.item()
                free_gpu([loss], delete=True)

                batch_count += 1
            pbar.update(1)

        if args.arch == 'bert-base-uncased_pt':
            if args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(),
                                               args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(),
                                               args.max_grad_norm)
        if args.finetune_epochs > 0:
            optim_e.step()
            if scheduler_e is not None:
                scheduler_e.step()
            optim_e.zero_grad()
        else:
            optim_e.step()
            if scheduler_e is not None:
                scheduler_e.step()
            optim_c.step()
            if scheduler_c is not None:
                scheduler_c.step()
            optim_e.zero_grad()

            # Experimenting with classifier accumulated gradient
            if args.replicate > 50:
                optim_c.zero_grad()

        epoch_losses.append(batch_loss)
        epoch_losses_contrastive.append(batch_loss_contrastive)
        epoch_losses_cross_entropy.append(batch_loss_cross_entropy)

        if (batch_ix + 1) % args.log_loss_interval == 0:
            print_output = f'Epoch {epoch:>3d} | Batch {batch_ix:>4d} | '
            print_output += f'Loss: {batch_loss:<.4f} (Epoch Avg: {np.mean(epoch_losses):<.4f}) | '
            print_output += f'CL: {batch_loss_contrastive:<.4f} (Epoch Avg: {np.mean(epoch_losses_contrastive):<.4f}) | '
            print_output += f'CE: {batch_loss_cross_entropy:<.4f}, (Epoch Avg: {np.mean(epoch_losses_cross_entropy):<.4f}) | '
            print_output += f'SW: {supervised_weight:<.4f}'
            print(print_output)

        if ((batch_ix + 1) % args.checkpoint_interval == 0 or
                (batch_ix + 1) == len(dataloader)):
            model = get_net(args)
            state_dict = encoder.to(torch.device('cpu')).state_dict()
            model = load_encoder_state_dict(model, state_dict)
            if 'bert' in args.arch:
                model.classifier = classifier
            else:
                model.fc = classifier
            checkpoint_name = save_checkpoint(model, None,
                                              np.mean(epoch_losses),
                                              epoch, batch_ix, args,
                                              replace=True,
                                              retrain_epoch=-1,
                                              identifier='fm')
            args.checkpoint_name = checkpoint_name

    epoch_losses = (epoch_losses,
                    epoch_losses_contrastive,
                    epoch_losses_cross_entropy)
    return encoder, classifier, epoch_losses


def compute_slice_outputs(erm_model, train_loader, test_criterion, args):
    """
    Compute predictions of ERM model to set up contrastive batches
    """
    if 'rep' in args.slice_with:
        slice_outputs = compute_slice_indices_by_rep(erm_model,
                                                     train_loader,
                                                     cluster_umap=True,
                                                     umap_components=2,
                                                     cluster_method=args.rep_cluster_method,
                                                     args=args,
                                                     visualize=True)
        sliced_data_indices, sliced_data_correct, sliced_data_losses = slice_outputs

    if 'pred' in args.slice_with:
        slice_outputs_ = compute_slice_indices(erm_model, train_loader,
                                               test_criterion, 1,
                                               args,
                                               resample_by='class',
                                               loss_factor=args.loss_factor,
                                               use_dataloader=True)
        sliced_data_indices_, sliced_data_losses_, sliced_data_correct_, sliced_data_probs_ = slice_outputs_

    if args.slice_with == 'pred_and_rep':
        # Combine the indices
        sliced_data_indices, sliced_data_correct = combine_data_indices(
            [sliced_data_indices, sliced_data_indices_],
            [sliced_data_correct, sliced_data_correct_])
    elif args.slice_with == 'pred':
        sliced_data_indices = sliced_data_indices_
        sliced_data_correct = sliced_data_correct_
        sliced_data_losses = sliced_data_losses_

    return sliced_data_indices, sliced_data_correct, sliced_data_losses


def sample_anchors(anchor_class, anchor_dict, num_anchor):
    p = None

    num_samples = num_anchor
    sample_indices = anchor_dict['ix_by_class'][anchor_class]
    replace = True if num_samples > len(sample_indices) else False
    sample_indices = np.random.choice(sample_indices,
                                      size=num_samples,
                                      replace=replace,
                                      p=p)
    return sample_indices


def sample_positives(anchor_class, positives_by_class, num_positive):
    positive_dict = positives_by_class[anchor_class]
    p = None
    num_samples = num_positive
    replace = True if num_samples > len(positive_dict['ix']) else False

    sample_indices = np.random.choice(np.arange(len(positive_dict['ix'])),
                                      size=num_samples,
                                      replace=replace,
                                      p=p)
    sample_slice_sources = positive_dict['source'][sample_indices]
    sample_indices = positive_dict['ix'][sample_indices]
    return sample_indices, sample_slice_sources


def sample_negatives(negative_dict, num_negative):
    p = None

    num_samples = num_negative
    replace = True if num_samples > len(negative_dict['ix']) else False
    sample_indices = np.random.choice(negative_dict['ix'],
                                      size=num_samples,
                                      replace=replace,
                                      p=p)
    return sample_indices


# Adjust number of negatives or positives if > sliced neg / pos
def adjust_num_pos_neg_(positives_by_class, slice_negatives,
                        args):

    print(f'given args: number of anchors: {args.num_anchor}')
    print(f'given args: number of positives: {args.num_positive}')
    print(f'given args: number of negatives: {args.num_negative}')
   
    num_pos = np.min([len(positives_by_class[c]['target'])
                      for c in range(args.n_cls)])
    num_neg = np.min([len(negative_dict['target'])
                      for negative_dict in slice_negatives])
    
    print(f'given samples: number of positives: {num_pos}')
    print(f'given samples: number of negatives: {num_neg}')
    
    num_pos = np.min((args.num_positive, num_pos))
    num_neg = np.min((args.num_negative, num_neg))
    
    
    # Tentative
    num_anc = np.min((args.num_anchor, np.min((num_pos, num_neg))))


    # Adjust arguments
    args.num_positive = num_pos
    args.num_negative = num_neg
    args.num_anchor = num_anc
    print(f'Adjusted number of anchors:   {args.num_anchor}')
    print(f'Adjusted number of positives: {args.num_positive}')
    print(f'Adjusted number of negatives: {args.num_negative}')
    
# Adjust number of anchors or hard negatives if > sliced anc / neg
def adjust_num_anc_neg_(slice_anchors, slice_negatives,
                        args):
    num_anc = np.min([len(anchor_dict['target'])
                      for anchor_dict in slice_anchors])
    num_neg = np.min([len(negative_dict['target'])
                      for negative_dict in slice_negatives])
    num_anc = np.min((args.num_anchor, num_anc))
    # num_neg Because now both anchors and negatives are from the nonspurious groups
    num_neg = np.min((args.num_negative_easy, num_anc))

    # Adjust experiment name to reflect
    # Adjust arguments
    args.num_anchor = num_anc
    args.num_negative_easy = num_neg
    print(f'Adjusted number of anchors:   {args.num_anchor}')
    print(f'Adjusted number of (hard) negatives: {args.num_negative_easy}')

def compute_slice_indices(opt, dataset):
    """
    Get "slices" of data belonging to different subgroups from the pre-extracted embeddings
    (cf. [data/embeddings_unnormalized/[waterbirds/celeba]/RN50/clip.json])

    Args:
    - dataset : Custom Dataset (cf. [[waterbirds/celeba]_embedding.py])
    Returns:
    - sliced_data_indices (int(np.array)[]): List of numpy arrays denoting indices of the dataloader.dataset
                                             corresponding to different slices 
    """
    # First compute pseudolabels
    
    # pseudo_labels = torch.hstack(all_predicted) # [N, ]
    # correct = torch.hstack(all_correct) # [N, 
    
    embeddings_df = dataset.embeddings_df
    # print(embeddings_df)
    # print(embeddings_df.index)
    # print(embeddings_df.columns)
    
    
    train_indices = (embeddings_df.loc["split"]==dataset.split_dict[dataset.split]).values
    train_embeddings_df = embeddings_df.T[train_indices]
    train_embeddings_df = train_embeddings_df.T
    
    pseudo_labels = train_embeddings_df.loc["y_pred"].values # [0, 1, 1, 0, 1....]
    if opt.dataset =='celeba':
        labels = train_embeddings_df.loc["blond"].values

    else:
        labels =  train_embeddings_df.loc["y"].values # [1, 1, 0, 0, 1, ....]
    correct = (pseudo_labels == labels) # [False, True, False, True, True, ...]
    
    sliced_data_indices = []
    all_correct = []
    for label in np.unique(pseudo_labels): # 0으로 예측
        group = np.where(pseudo_labels == label)[0] # [0, 3, ...] / [1, 2, 4, ...]
        correct_by_group = correct[group] # [False, True, ...] / [True, False, True, ...]
        
        sliced_data_indices.append(group) # 
        print(group)
        all_correct.append(correct_by_group) 
    
    
    return sliced_data_indices, all_correct # [[0, 3,...], [1, 2, 4, ...]], [[False, True, ...], [True, False, True, ...]]


def prepare_contrastive_points(dataset, sliced_data_indices,
                               sliced_data_correct,
                               ):
    train_targets = dataset.y_array
    train_spurious = dataset.confounder_array
    sliced_data_indices_all = np.concatenate(sliced_data_indices)
    sliced_data_correct_all = np.zeros(len(train_targets))
    sliced_data_correct_all[sliced_data_indices_all] = np.concatenate(
        sliced_data_correct)
    
    sliced_data_incorrect = []
    for slice_ix, boolean_array in enumerate(sliced_data_correct):
        sliced_data_incorrect.append(np.array([not bool for bool in boolean_array]))
        
    sliced_data_incorrect = np.array(sliced_data_incorrect)
    
    all_anchors = {'slice_ix': np.zeros(len(train_targets)).astype(int), # 4765
                   'in_slice_ix': np.zeros(len(train_targets)).astype(int)} # 4765

    # Store all anchors and negatives
    slice_anchors = [None] * len(sliced_data_indices)
    slice_negatives = [None] * len(sliced_data_indices)
    
    # For positives, just specify by the ground-truth NOTE No.
    # (These are the same as negatives in another slice, just organized by class) 
     ## another slice : 1 prediction. 0 class (즉, 그냥 틀린 친구들 in CnC)
     
    # Cnc : Anchor -> correct 
      # Neg : Different class & Same Prediction
      # Pos : Different prediction & Same class
    # CA : Anchor -> incorrect
    
    positives_by_class = {}
    for slice_ix, data_indices in enumerate(sliced_data_indices): # slice_ix = 0 (즉, prediction=0일 때 기준 서술)
        
        target_class, target_counts = np.unique(train_targets[data_indices],
                                                return_counts=True)
        
        for tc_ix, tc in enumerate(target_class):
            print(f'>> Slice {slice_ix}, target: {tc}, counts: {target_counts[tc_ix]}')
        
        # Anchors are datapoints in the slice that the model got in-correct (False)
        ix = np.where(sliced_data_incorrect[slice_ix])[0] # prediction 0, class 1
        print(
            f'Slice {slice_ix} % incorrect: {len(ix) / len(data_indices) * 100:<.4f} %')

        slice_ix_anchors = {'ix': data_indices[ix],
                            'target': train_targets[data_indices][ix], # Only 1
                            'incorrect': sliced_data_incorrect[slice_ix][ix], # all True
                            'source': np.ones(len(data_indices[ix])).astype(int) * slice_ix, # zero_prediction
                            'spurious': train_spurious[data_indices][ix], # 0 or 1 (다만 1이 그 전체 양(5%)에 비해선 비교적 많을 것)
                            'ix_by_class': {},} # 1: data_indices[ix] (Class 1)
        # Zeroshot prediction 값([0/1])에 따른 data indices 들에 대한 정보들
        
        # anchor: prediction 0 -> class 1 (즉, zero-prediction에 대한 anchor는 모두 class 1) -> indices
        for t in np.unique(train_targets[data_indices][ix]):
            tix = np.where(train_targets[data_indices][ix] == t)[0]
            slice_ix_anchors['ix_by_class'][t] = data_indices[ix][tix] 
            
        # Negatives: prediction 0 -> class 0 (즉, Anchor와 Different Class) -> indices
        ## 이 중에 가까운 샘플만 골라서 사용(Water birds에서는 거의 대부분 사용한다고 봐도 무방하다) 
        nix = np.setdiff1d(np.arange(len(data_indices)), ix) # prediction 0, class 0 (True)
        
        target_class, target_counts = np.unique(train_targets[data_indices][ix], # Anchor와 같은 Class
                                                return_counts=True) # (1, 254) (Class 1, anchor 개수)
        print("ix class counts", target_class, target_counts)
        target_class, target_counts = np.unique(train_targets[data_indices][nix], # Anchor와 같은 Class
                                                return_counts=True) # (1, 3588) (Class 0, negative samples 중 일부)
        print("nix class counts", target_class, target_counts)
        
        print(f'Slice {slice_ix} # negative (correct): {len(nix)}')
        print(
            f'Slice {slice_ix} % negative (correct): {len(nix) / len(data_indices) * 100 :<.4f} %')
        
        print(
            f'Unique negative targets: {np.unique(train_targets[data_indices][nix], return_counts=True)}')

        slice_ix_negatives = {'ix': list(data_indices[nix]),
                                'target': list(train_targets[data_indices][nix]), # All 0
                                'incorrect': list(sliced_data_incorrect[slice_ix][nix]), # All False (맞았으므로)
                                'source': list(np.ones(len(data_indices[nix])).astype(int) * slice_ix), # All 0 Prediction 0
                                'spurious': list(train_spurious[data_indices][nix])} # 0이 많을 것(Major group) (True)

        
        
        
        # Positives: "Different prediction" and "Same Class(True)"
        ## 즉, 0 prediction & 0 class (다른 Slice의 Positive다)
        # Positives, for other slices - for here just save by unique class that was also "correct"
        ## 즉, 
        target_class, target_counts = np.unique(train_targets[data_indices][nix], # nix : correct
                                                return_counts=True) # (0, 3588) (Class 0, # Correct sample )
        
        print("nix class counts(for positive)", target_class, target_counts)
        
        correct_data_indices = data_indices[nix] # Pred 0 / Class 0
        
        print(f"Slice {slice_ix} # Positive: (for 'other' slice)", len(correct_data_indices))
        
        
        for cix, c in enumerate(target_class): # only 0

            pix = np.where(train_targets[correct_data_indices] == c)[0]   

            pos_data_indices = list(correct_data_indices[pix])
            pos_data_targets = list(
                train_targets[correct_data_indices][pix])
            pos_data_correct = list(
                sliced_data_correct[slice_ix][nix][pix])
            pos_data_source = list(
                np.ones(len(data_indices[nix][pix])).astype(int) * slice_ix)
            pos_data_spurious = list(
                train_spurious[correct_data_indices][pix])
            if c in positives_by_class:
                positives_by_class[c]['ix'].extend(pos_data_indices)
                positives_by_class[c]['target'].extend(pos_data_targets)
                positives_by_class[c]['correct'].extend(pos_data_correct)
                positives_by_class[c]['source'].extend(pos_data_source)
                positives_by_class[c]['spurious'].extend(pos_data_spurious)
            else:
                positives_by_class[c] = {'ix': pos_data_indices,
                                            'target': pos_data_targets,
                                            'correct': pos_data_correct,
                                            'source': pos_data_source,
                                            'spurious': pos_data_spurious}

        
        # Save
        slice_anchors[slice_ix] = slice_ix_anchors
        slice_negatives[slice_ix] = slice_ix_negatives
    
    # NOTE Add Easy-Negatives samples (일단 잘 안되니 추가해보자 ㅠㅠ) (add negative:  Differennt class & Different prediction) (원래 Different class & Same prediction만)
    for slice_ix, data_indices in enumerate(sliced_data_indices): # slice_ix = 0 (즉, prediction=0일 때 기준 서술)
        print("> Add Easy Negatives samples (0523)")
        target_class, target_counts = np.unique(train_targets[data_indices],
                                                return_counts=True)
        
        another_slice_ix = np.abs(slice_ix - 1)
        # for tc_ix, tc in enumerate(target_class):
        #     print(f'>> Slice {slice_ix}, target: {tc}, counts: {target_counts[tc_ix]}')
        
        # Anchors are datapoints in the slice that the model got in-correct (False)
        ix = np.where(sliced_data_incorrect[slice_ix])[0] # prediction 0, class 1
        print(
            f'>> % Negatives for Anoter Slice {another_slice_ix}): {len(ix) / len(data_indices) * 100:<.4f} %')
        print(f"{len(slice_negatives[another_slice_ix]['ix'])} -> {len(slice_negatives[another_slice_ix]['ix']) + len(data_indices[ix])}")
        
        # print("unique(source): ", np.unique(slice_negatives[another_slice_ix]['source'], return_counts=True))
        slice_negatives[another_slice_ix]['ix'].extend(data_indices[ix])
        slice_negatives[another_slice_ix]['target'].extend(train_targets[data_indices][ix]) # 1
        slice_negatives[another_slice_ix]['incorrect'].extend(sliced_data_incorrect[slice_ix][ix]) # 원래 다 False였는데 True도 포함될 것.
        slice_negatives[another_slice_ix]['source'].extend(list(np.ones(len(data_indices[ix])).astype(int) * slice_ix))
        slice_negatives[another_slice_ix]['spurious'].extend(list(train_spurious[data_indices][ix]))
        
    
    
    # # Fill in positives if no slices had the class as spurious
    # for slice_ix, data_indices in enumerate(sliced_data_indices):
    #     target_class, target_counts = np.unique(train_targets[data_indices],
    #                                             return_counts=True)

    #     # Compare average correctness, still use the max_class variable
    #     avg_correct = []
    #     for c in target_class:
    #         class_indices = np.where(train_targets[data_indices] == c)[0]
    #         class_correct = sliced_data_correct[slice_ix][class_indices]
    #         avg_correct.append(np.mean(class_correct))
    #     max_class_ix = np.argmax(avg_correct)
        
    #     for c in target_class:
    #         if c not in positives_by_class:
    #             print("asdsdfksdlfkjsdflksajflk;sadfjlsakdfj;alsdkfjsdal;kfjalks;fjl;ksjdflks")
    #             print(
    #                 f'> Loading correct datapoints as positives for class {c} from slice {slice_ix}')
    #             ix = np.where(train_targets[data_indices] == c)[0]
    #             positives_by_class[c] = {'ix': list(data_indices[ix]),
    #                                      'target': list(train_targets[data_indices][ix]),
    #                                      'correct': list(sliced_data_correct[slice_ix][ix]),
    #                                      'source': list(np.ones(len(data_indices[ix])).astype(int) * slice_ix),
    #                                      'spurious': list(train_spurious[data_indices][ix])}

    # Convert casted lists back to ndarrays
    for c, class_dict in positives_by_class.items():
        for k, v in class_dict.items():
            positives_by_class[c][k] = np.array(v)

    for ix, slice_negative in enumerate(slice_negatives):
        for k, v in slice_negative.items():
            slice_negatives[ix][k] = np.array(v)


    return slice_anchors, slice_negatives, positives_by_class, all_anchors


def construct_contrastive_data(slice_anchors, slice_negatives, positives_by_class, args): # Data processing 오류 핸들링.
    # Get number of negatives per target class
    args.num_negatives_by_target = [0] * args.n_cls

    batch_samples = []
    batch_samples_old = []
    
    def sample_anchors(anchor_class, anchor_dict, num_anchor):
        p = None

        num_samples = num_anchor
        sample_indices = anchor_dict['ix_by_class'][anchor_class]
        replace = True if num_samples > len(sample_indices) else False
        sample_indices = np.random.choice(sample_indices,
                                        size=num_samples,
                                        replace=replace,
                                        p=p)
        return sample_indices


    def sample_positives(anchor_class, positives_by_class, num_positive):
        positive_dict = positives_by_class[anchor_class]
        p = None
        num_samples = num_positive
        replace = True if num_samples > len(positive_dict['ix']) else False
        sample_indices = np.random.choice(np.arange(len(positive_dict['ix'])),
                                        size=num_samples,
                                        replace=replace,
                                        p=p)
        sample_slice_sources = positive_dict['source'][sample_indices]
        sample_indices = positive_dict['ix'][sample_indices]
        return sample_indices, sample_slice_sources


    def sample_negatives(negative_dict, num_negative):
        p = None

        
        num_samples = num_negative
        replace = True if num_samples > len(negative_dict['ix']) else False
        sample_indices = np.random.choice(negative_dict['ix'],
                                        size=num_samples,
                                        replace=replace,
                                        p=p)
        return sample_indices
    
    for slice_ix, anchor_dict in enumerate(slice_anchors):
        # Prediction 0.
        batch_samples_per_slice = []  # First aggregate within
        negative_dict = slice_negatives[slice_ix] # Prediction 0, Class 0 
        
        # For hard negative
        args.num_negatives_by_target[slice_ix] = len(negative_dict['ix']) #  [[0],[0]] #->  [[3588], [0]] -> [[3588], [859]]

        anchor_targets = anchor_dict['target'] # All 1
        anchor_indices = anchor_dict['ix'] # Anchor 254개 
        
        # 254, 94 (Prediction 0에서의 False, Prediction 1에서의 False )
        for aix, anchor_ix in enumerate(tqdm(anchor_indices, desc=f'Generating data from slice {slice_ix}')): 

            anchor_class = anchor_targets[aix]
            
            # Sample additional positives 
            anchor_indices = sample_anchors(anchor_class,
                                            anchor_dict,
                                            args.num_anchor - 1)

            anchor_indices = np.concatenate([[anchor_ix], anchor_indices]) # (1, )

            positive_outputs = sample_positives(anchor_class,
                                                positives_by_class,
                                                args.num_positive)
            
            positive_indices, positive_slice_sources = positive_outputs # (n_positives, )


            
            # Keep as this, in case want to generate new neg per pos as before
            samples = [anchor_indices, positive_indices]
            negative_indices = sample_negatives(negative_dict,
                                                args.num_negative) # (n_negatives, )
            samples.append(negative_indices)


            
            batch_sample = np.concatenate(samples) # (# positive * # negative)
            batch_samples_per_slice.append(batch_sample)
            batch_samples_old.append(batch_sample)
            
        np.random.shuffle(batch_samples_per_slice) # (# anchor, # positive * # negative): (254, 1719) / (94. 1719)
        batch_samples.append(batch_samples_per_slice)
        print("batch_samples_per_slice.shape", np.array(batch_samples_per_slice).shape) 
    
    return batch_samples

def load_contrastive_loader(trainset, batch_samples, args,
                          persistent_workers=True): # Data processing 오류 핸들링.
    # Batch samples [(254, 1719), (94, 1719)] : Sliced batch samples (Original)
    
    # 임시
    # for slice in range(len(batch_samples)):
    #     batch_samples[slice] = list(np.array(batch_samples[slice])[:10-slice*7, :5])
    
    # Shuffle for "selecting "
    
    # Balancing By Class (downsampling the class with more zero-shot falilure samples) ([Water-birds/Land-background] in Waterbirds datasets )
    if args.balance_by_zs_pred:    
        if args.re_shuffle_ca_loader:
            print("Shuffle for anchor of majority class(0 or 1) (Just shuffling all the class)") # Minority class에 해당하는 sample은 어차피 모두 포함됨.
            for slice in range(len(batch_samples)):

                np.random.shuffle(batch_samples[slice]) 

        print("Balancing contrastive samples [by class (down-sample class with more zero-shot failure samples)]")
        batch_samples = list(zip(*batch_samples)) # (94, 2, 1719) (우선 Prediction 비율에 따라 Balanced로 구성.) #NOTE
        batch_samples = np.array(batch_samples)
        batch_samples = batch_samples.reshape(-1, batch_samples.shape[-1]) # (188, 1719) # Pos -> Neg -> Pos -> Neg .... 
        if (not args.maintain_alternative_ordering) and (args.re_shuffle_ca_loader) : # Reshuffle contrastive samples # 선택 ([pos, neg, pos,neg, pos, neg] -> [pos, pos, neg, pos, neg, neg])
            print("Not maintaining alternative ordering, after balancing contrastive samples by class. ")
            np.random.shuffle(batch_samples)  # Re-shuffle After concatenating
    else:        
        print("No balancing contrastive samples (Focusing on class with more prediction errors)")
        batch_samples = np.concatenate(batch_samples) # (348, 1719) (우선 Prediction 비율에 따라 Balanced로 구성.) #NOTE    
        
        if args.re_shuffle_ca_loader:
            print("Shuffle for concatenated-contrastive-batch")
            np.random.shuffle(batch_samples)  # Re-shuffle After concatenating # 필수
    
    print("First 10 Anchor indices: ", batch_samples[:10, 0])
    print("Shape of Contrastive Batch Samples : ", batch_samples.shape)
    contrastive_indices = np.concatenate(batch_samples)
    
    contrastive_train_set = get_resampled_set(trainset,
                                              contrastive_indices,
                                              copy_dataset=True)

    contrastive_dataloader = DataLoader(contrastive_train_set,
                                        batch_size=len(
                                            batch_samples[0]) * int(args.batch_factor),
                                        shuffle=False, num_workers=args.num_workers, persistent_workers = persistent_workers)
    print("> batchsize of contrastive data loader :", len(batch_samples[0]) * int(args.batch_factor)) # (32) * (1719)[1 + pos + neg]
    print("> len of contrastive dataset :", len(contrastive_train_set)) # (188*1719, )
    return contrastive_dataloader

def get_resampled_set(dataset, resampled_set_indices, copy_dataset=False):
    """
    Obtain spurious dataset resampled_set
    Args:
    - dataset (torch.utils.data.Dataset): Spurious correlations dataset
    - resampled_set_indices (int[]): List-like of indices 
    - deepcopy (bool): If true, copy the dataset
    """
    resampled_set = copy.deepcopy(dataset) if copy_dataset else dataset
    try:  # Some dataset classes may not have these attributes (WaterbirdsEmbeddingsDatasets 특화)
        resampled_set.y_array = resampled_set.y_array[resampled_set_indices]
        resampled_set.group_array = resampled_set.group_array[resampled_set_indices]
        resampled_set.split_array = resampled_set.split_array[resampled_set_indices]
        resampled_set.targets = resampled_set.targets[resampled_set_indices]
        resampled_set.targets_group = resampled_set.targets_group[resampled_set_indices]
        resampled_set.targets_spurious = resampled_set.targets_spurious[resampled_set_indices]
        try:  # Depending on the dataset these are responsible for the X features
            resampled_set.filename_array = resampled_set.filename_array[resampled_set_indices]
        except:
            resampled_set.x_array = resampled_set.x_array[resampled_set_indices]
    except AttributeError as e:
        print(e)

    # Main Embedding Information을 담고있는 Dataframe 조절 (6, 11760) -> (6, 323171...)
    resampled_embeddings_df = resampled_set.embeddings_df.copy()
    resampled_embeddings_df = resampled_embeddings_df.T
    resampled_embeddings_df = resampled_embeddings_df.iloc[resampled_set_indices]
    resampled_set.embeddings_df = resampled_embeddings_df.T.copy()
    
    # Indexing 방식 수정 (dataframe+key -> dataframe+iloc)
    resampled_set.on_contrastive_batch = True

    print('len(resampled_set.targets)', len(resampled_set.targets))
    return resampled_set


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(SupervisedContrastiveLoss, self).__init__()
        self.cl_temperature = args.cl_temperature
        self.n_positives = args.num_positive
        self.n_negatives = args.num_negative
        self.args = args
    
        self.sim = nn.CosineSimilarity(dim=1)
        
    def forward(self, model, contrastive_batch):
        
        # contrastive_batch [anc; pos; neg] : (1+N+M(=N), )
        # Compute negative similarities
        neg_indices = [0] + list(range(len(contrastive_batch))[
            -self.n_negatives:])
        anchor_negatives = contrastive_batch[neg_indices]
        
        # Compute positive similarities
        anchor_positives = contrastive_batch[:1 + self.n_positives]
        pos = self.compute_sim(model, anchor_positives, 
                                       return_sum=False)
        max_pos, _ = torch.max(pos, dim=0, keepdim=True)
        pos = pos - max_pos.detach() # 그래디언트 흐를듯
        exp_pos = torch.exp(pos)
        anchor_positives = anchor_positives.to(torch.device('cpu'))
        
        # M개의 exp(sim) score
        neg = self.compute_sim(model, anchor_negatives,
                                       return_sum=False)
        
        neg = neg - max_pos.detach() # 같은 Scaling 먹여야.
        exp_neg = torch.exp(neg)
        
        anchor_negatives = anchor_negatives.to(torch.device('cpu'))
        
        sum_exp_neg = exp_neg.sum(0, keepdim=True)
        pos_numerator = exp_pos.mean()
        neg_numerator = exp_neg.mean()
        denominator = sum_exp_neg + exp_pos.sum(0, keepdim=True)
        log_probs = (torch.log(exp_pos) - 
                        torch.log(sum_exp_neg + exp_pos.sum(0, keepdim=True)))
        # print("pos 분자 :", exp_pos.shape)
        # print("pos 분모 :", exp_pos.shape)
        # print("neg 분모 :", exp_neg.shape)
        loss = -1 * log_probs
        del exp_pos; del exp_neg; del log_probs
        
        # return loss.mean() # N개의 Positives에 대한 평균. 
        return loss.mean(), pos_numerator, neg_numerator, denominator # NOTE 학습 추이 관찰용. 
    
    def compute_sim(self, model, features, return_sum=True):
        """
        Compute sum(sim(anchor, pos)) or sum(sim(anchor, neg))
        First index : anchor
        """
        # in Contrastive Adapter, features:CLIP, outputs:Adapted-CLIP 
        
        if self.args.tl_method =="contrastive_adapter":
            outputs = model.forward_ca(features) # normalized: model.forward_ca // (normalized)logits까지: modal.forward // unnormalized: model.adapter - 
        
        sim = self.sim(outputs[0].view(1, -1), outputs[1:]) # (N,) or (M,)
        sim_divided_by_temp = torch.div(sim, self.cl_temperature)
        
        outputs = outputs.to(torch.device('cpu'))
        return sim_divided_by_temp

def skim_dataloader_by_group(dl, max_batch = 10):
    print('Distribution of some mini batch ')
    total_num_images = len(dl.dataset)
    idxs_seen = []
    class_0_batch_counts = []
    class_1_batch_counts = []
    class_2_batch_counts = []
    class_3_batch_counts = []

    for i, batch in enumerate(dl):
        _, all_labels, _ = batch
        
        classes = all_labels['group']
        class_ids, class_counts = classes.unique(return_counts=True)
        class_ids = set(class_ids.tolist())
        class_counts = class_counts.tolist()

                
        try:
            print(f"[{class_counts[0]}, {class_counts[1]}, {class_counts[2]}, {class_counts[3]}]")
        except:
            print(f"[{class_counts[0]}, {class_counts[1]}, {class_counts[2]}]")
            
            
        
        if i==max_batch:
            break

def GetNegativesByClass(opt, trainset, positives_by_class):

    embeddings_df = trainset.embeddings_df
    train_indices = (embeddings_df.loc["split"]==trainset.split_dict[trainset.split]).values
    train_embeddings_df = embeddings_df.T[train_indices]
    train_embeddings_df = train_embeddings_df.T
    
    if opt.dataset =="celeba":
        labels =  train_embeddings_df.loc["blond"].values # [1, 1, 0, 0, 1, ....]
    else:
        labels =  train_embeddings_df.loc["y"].values # [1, 1, 0, 0, 1, ....]
    negatives_by_class = {}
    for c in range(trainset.n_classes):
        print(f"> Class {c}")
        negatives_by_class[c] = {}
        class_indices = np.where(c==labels)[0]
        print(f"# of samples :", len(class_indices))
        print("# of positives : ", len(positives_by_class[c]['ix']))

        negatives_by_class[c]['ix'] = np.setdiff1d(class_indices, positives_by_class[c]['ix'])
        print("# of negatives : ", len(negatives_by_class[c]['ix']))

    return negatives_by_class


def GetResampledWeightsCE(dataset, positives_by_class, negatives_by_class, opt):
    weights = np.ones(len(dataset))
    print("Re-sampling for [Cross Entropy Loader]")
    
    
    stat_for_correct = {}
    for c in range(dataset.n_classes):
        stat_for_correct[c] = {}
        n_pos = len(positives_by_class[c]['ix'])
        n_neg = len(negatives_by_class[c]['ix'])
        
        stat_for_correct[c]['num_cls'] = n_pos + n_neg
        stat_for_correct[c]['num_pos'] = n_pos
        if n_pos < n_neg: # 이럴 일은 거의 없겠다만, Sampling X
            weights[positives_by_class[c]['ix']] = 1
            weights[negatives_by_class[c]['ix']] = 1
        else: # # of negatives -> # of positives (oversampling)
            upsample_weight = n_pos / n_neg
            print(f"> Class {c} Weight: {upsample_weight :.3f}")
            weights[positives_by_class[c]['ix']] = 1
            weights[negatives_by_class[c]['ix']] = upsample_weight
        
        
    # Class-distribution bias-correction -> only possible in binary classifiation (일단))
    # 쓸 수 있는 정보는 우선 쓰자. 
    if opt.correct_class_bias or opt.reweighting_by_class:
        print("Correction bias of class-distribution (caused by [up-sampling-zero-shot-failure] strategy)")
        if stat_for_correct[0]['num_cls'] < stat_for_correct[1]['num_cls']:
            major_c = 1; minor_c = 0
            imbal_c = stat_for_correct[1]['num_cls'] / stat_for_correct[0]['num_cls']
            reweighted_imbal_c = stat_for_correct[1]['num_pos'] / stat_for_correct[0]['num_pos']
        else:
            major_c = 0; minor_c = 1
            imbal_c = stat_for_correct[0]['num_cls'] / stat_for_correct[1]['num_cls']
            reweighted_imbal_c = stat_for_correct[0]['num_pos'] / stat_for_correct[1]['num_pos']
        
        print(f">Original class imbalance weight (majority class : {major_c}) :", imbal_c)
        print(f">Re-weighted (biased) class imbalance weight ", reweighted_imbal_c)
    
        if imbal_c < reweighted_imbal_c:
            if not opt.reweighting_by_class:
                print(f">>Correction for minority class {minor_c} :", reweighted_imbal_c / imbal_c)
                minor_indices = np.concatenate([positives_by_class[minor_c]['ix'], negatives_by_class[minor_c]['ix']])
                weights[minor_indices] = weights[minor_indices] * (reweighted_imbal_c / imbal_c)
            else:    
                print(f">>Balancing for minority class {minor_c} :", reweighted_imbal_c / 1 )
                minor_indices = np.concatenate([positives_by_class[minor_c]['ix'], negatives_by_class[minor_c]['ix']])
                weights[minor_indices] = weights[minor_indices] * (reweighted_imbal_c / 1)
        else:
            if opt.reweighting_by_class:
                print(f">>Balancing for minority class {minor_c} :", reweighted_imbal_c / 1)
                minor_indices = np.concatenate([positives_by_class[minor_c]['ix'], negatives_by_class[minor_c]['ix']])
                weights[minor_indices] = weights[minor_indices] * (reweighted_imbal_c / 1)
            else:
                pass
        
        
    print("> Final sampling weights set :", np.unique(weights, return_counts=True)[0])
    print(">> Corresponding Counts :", np.unique(weights, return_counts=True)[1], f"/ {len(dataset)}" )
    
    
    return weights

# def initialize_for_vis(opt):
#     best_acc = 0
#     best_epoch = 0
#     best_model = None
#     # opt = parse_option()

#     print(f"> Start Transfer Learning using [{opt.tl_method}]")
#     print('========================================================================')
#     if opt.dataset == 'waterbirds':
#         # build dataset example.
#         print(f"Load image embedding of Waterbirds: {opt.image_embedding_dir}")
#         trainset = WaterbirdsEmbeddings(opt.data_dir, 'train', opt.image_embedding_dir, None)
#         print(f"ㄴ Corresponding text embedding of Waterbirds: {opt.text_embedding_dir}")
#         # build data loader
#         print("Load Data Loader (train, validation, test)")
#         train_loader, val_loader, test_loader = load_waterbirds_embeddings(opt.data_dir, opt.image_embedding_dir, opt.batch_size, opt.batch_size)
        
#         # print training target
#         if opt.train_target == "class":
#             print(f"Training target : {opt.train_target} (Land bird(0) / Water bird(1))")
#         elif opt.train_target == "spurious":
#             print(f"Training target : {opt.train_target} (Land background(0) / Water background(1))")
        
#     elif opt.dataset == 'celeba':
#         # build dataset example.
#         print(f"Load embedding of CelebA: {opt.image_embedding_dir}")
#         trainset = CelebaEmbeddings(opt.data_dir, 'train', opt.image_embedding_dir, None)
#         print(f"ㄴ Corresponding text embedding of Waterbirds: {opt.text_embedding_dir}")
#         # build data loader
#         print("Load Data Loader (train, validation, test)")
#         train_loader, val_loader, test_loader = load_celeba_embeddings(opt.data_dir, opt.image_embedding_dir, opt.batch_size, opt.batch_size)
        
#         # print training target
#         if opt.train_target == "class":
#             print(f"Training target : {opt.train_target} (non-blond hair(0) / blond hair(1))")
#         elif opt.train_target == "spurious":
#             print(f"Training target : {opt.train_target} (female(0) / male(1))")

#     # group information
#     get_yp_func = partial(get_y_p, n_places=trainset.n_places)
#     train_group_ratio = trainset.group_ratio

#     # build model and criterion
#     classifier, criterion = set_model(opt) # model, 
    
    
#     return trainset, train_loader, val_loader, test_loader, get_yp_func, train_group_ratio, classifier, criterion

def initialize_for_vis(opt):
    best_acc = 0
    best_epoch = 0
    best_model = None
    # opt = parse_option()
    
    
    print(f"> Start Transfer Learning using [{opt.tl_method}]")
    print('========================================================================')
    if opt.dataset == 'waterbirds':
        # build data loader
        # if opt.tl_method in ["adapter_reg", "adapter_reg_seq", 'adapter_reg_seq_alter'] :
        #     from data.waterbirds_embeddings_reg import WaterbirdsEmbeddings, load_waterbirds_embeddings
        #     print(f"Load image embedding of Waterbirds: {opt.image_embedding_dir}")
        #     trainset = WaterbirdsEmbeddings(opt.data_dir, 'train', opt.image_embedding_dir, None)
        #     print("Load Data Loader (train, validation, test)")
        #     train_loader, reg_loader, val_loader, test_loader = load_waterbirds_embeddings(opt.data_dir, opt.image_embedding_dir, opt.batch_size, opt.batch_size_reg)
        # else:
        from data.waterbirds_embeddings import WaterbirdsEmbeddings, load_waterbirds_embeddings
        print(f"Load image embedding of Waterbirds: {opt.image_embedding_dir}")
        trainset = WaterbirdsEmbeddings(opt.data_dir, 'train', opt.image_embedding_dir, None)
        print(f"ㄴ Corresponding text embedding of Waterbirds: {opt.text_embedding_dir}")
        train_loader, val_loader, test_loader = load_waterbirds_embeddings(opt.data_dir, opt.image_embedding_dir, opt.batch_size, opt.batch_size)
        
        if opt.train_target == "class":
            print(f"Training target : {opt.train_target} (Land bird(0) / Water bird(1))")
        elif opt.train_target == "spurious":
            print(f"Training target : {opt.train_target} (Land background(0) / Water background(1))")
        
    elif opt.dataset == 'celeba':
        # build dataset example.
        from data.celeba_embeddings import CelebaEmbeddings, load_celeba_embeddings # 버근가.. 왜 인식을 몬하지
        print(f"Load embedding of CelebA: {opt.image_embedding_dir}")
        trainset = CelebaEmbeddings(opt.data_dir, 'train', opt.image_embedding_dir, None)
        
        # build data loader
        print("Load Data Loader (train, validation, test)")
        
        # train_loader, val_loader, test_loader = load_celeba_embeddings(opt.data_dir, opt.image_embedding_dir, opt.batch_size, opt.batch_size)
        # if opt.tl_method in ["adapter_reg", "adapter_reg_seq", 'adapter_reg_seq_alter'] :
        #     from data.celeba_embeddings_reg import CelebaEmbeddings, load_celeba_embeddings
        #     print(f"Load embedding of CelebA: {opt.image_embedding_dir}")
        #     trainset = CelebaEmbeddings(opt.data_dir, 'train', opt.image_embedding_dir, None)
        #     print("Load Data Loader (train, validation, test)")
        #     train_loader, reg_loader, val_loader, test_loader = load_celeba_embeddings(opt.data_dir, opt.image_embedding_dir, opt.batch_size, opt.batch_size_reg)
        # else:
        print(f"Load image embedding of Waterbirds: {opt.image_embedding_dir}")
        trainset = CelebaEmbeddings(opt.data_dir, 'train', opt.image_embedding_dir, None)
        print(f"ㄴ Corresponding text embedding of Waterbirds: {opt.text_embedding_dir}")
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
 
    return trainset, train_loader, val_loader, test_loader, get_yp_func, train_group_ratio, classifier, criterion
 
# def GetNegativesByClass(trainset, positives_by_class):

#     embeddings_df = trainset.embeddings_df
#     train_indices = (embeddings_df.loc["split"]==trainset.split_dict[trainset.split]).values
#     train_embeddings_df = embeddings_df.T[train_indices]
#     train_embeddings_df = train_embeddings_df.T
    
#     labels =  train_embeddings_df.loc["y"].values # [1, 1, 0, 0, 1, ....]
    
#     negatives_by_class = {}
#     for c in range(trainset.n_classes):
#         print(f"> Class {c}")
#         negatives_by_class[c] = {}
#         class_indices = np.where(c==labels)[0]
#         print(f"# of samples :", len(class_indices))
#         print("# of positives : ", len(positives_by_class[c]['ix']))

#         negatives_by_class[c]['ix'] = np.setdiff1d(class_indices, positives_by_class[c]['ix'])
#         print("# of negatives : ", len(negatives_by_class[c]['ix']))

#     return negatives_by_class


def GetResampledWeightsCE(dataset, positives_by_class, negatives_by_class, opt):
    weights = np.ones(len(dataset))
    print("Re-sampling for [Cross Entropy Loader]")
    
    
    stat_for_correct = {}
    for c in range(dataset.n_classes):
        stat_for_correct[c] = {}
        n_pos = len(positives_by_class[c]['ix'])
        n_neg = len(negatives_by_class[c]['ix'])
        
        stat_for_correct[c]['num_cls'] = n_pos + n_neg
        stat_for_correct[c]['num_pos'] = n_pos
        if n_pos < n_neg: # 이럴 일은 거의 없겠다만, Sampling X
            weights[positives_by_class[c]['ix']] = 1
            weights[negatives_by_class[c]['ix']] = 1
        else: # # of negatives -> # of positives (oversampling)
            upsample_weight = n_pos / n_neg
            print(f"> Class {c} Weight: {upsample_weight :.3f}")
            weights[positives_by_class[c]['ix']] = 1
            weights[negatives_by_class[c]['ix']] = upsample_weight
        
        
    # Class-distribution bias-correction -> only possible in binary classifiation (일단))
    # 쓸 수 있는 정보는 우선 쓰자. 
    if opt.correct_class_bias or opt.reweighting_by_class:
        print("Correction bias of class-distribution (caused by [up-sampling-zero-shot-failure] strategy)")
        if stat_for_correct[0]['num_cls'] < stat_for_correct[1]['num_cls']:
            major_c = 1; minor_c = 0
            imbal_c = stat_for_correct[1]['num_cls'] / stat_for_correct[0]['num_cls']
            reweighted_imbal_c = stat_for_correct[1]['num_pos'] / stat_for_correct[0]['num_pos']
        else:
            major_c = 0; minor_c = 1
            imbal_c = stat_for_correct[0]['num_cls'] / stat_for_correct[1]['num_cls']
            reweighted_imbal_c = stat_for_correct[0]['num_pos'] / stat_for_correct[1]['num_pos']
        
        print(f">Original class imbalance weight (majority class : {major_c}) :", imbal_c)
        print(f">Re-weighted (biased) class imbalance weight ", reweighted_imbal_c)
    
        if imbal_c < reweighted_imbal_c:
            if not opt.reweighting_by_class:
                print(f">>Correction for minority class {minor_c} :", reweighted_imbal_c / imbal_c)
                minor_indices = np.concatenate([positives_by_class[minor_c]['ix'], negatives_by_class[minor_c]['ix']])
                weights[minor_indices] = weights[minor_indices] * (reweighted_imbal_c / imbal_c)
            else:    
                print(f">>Balancing for minority class {minor_c} :", reweighted_imbal_c / 1 )
                minor_indices = np.concatenate([positives_by_class[minor_c]['ix'], negatives_by_class[minor_c]['ix']])
                weights[minor_indices] = weights[minor_indices] * (reweighted_imbal_c / 1)
        else:
            if opt.reweighting_by_class:
                print(f">>Balancing for minority class {minor_c} :", reweighted_imbal_c / 1)
                minor_indices = np.concatenate([positives_by_class[minor_c]['ix'], negatives_by_class[minor_c]['ix']])
                weights[minor_indices] = weights[minor_indices] * (reweighted_imbal_c / 1)
            else:
                pass
        
        
    print("> Final sampling weights set :", np.unique(weights, return_counts=True)[0])
    print(">> Corresponding Counts :", np.unique(weights, return_counts=True)[1], f"/ {len(dataset)}" )
    
    
    return weights