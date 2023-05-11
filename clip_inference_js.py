import argparse
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import clip

from sklearn.metrics import classification_report
from tqdm import tqdm

from data.celeba import CelebA
from data.waterbirds_js import Waterbirds
import classic_templates
import classic_celeba_templates
import classic_waterbirds_templates

from collections import defaultdict

def main(args):
    model, preprocess = clip.load(args.backbone, device='cuda', jit=False)  # RN50, RN101, RN50x4, ViT-B/32

    crop = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])
    transform = transforms.Compose([crop, preprocess])

    if args.dataset == 'waterbirds':
        template = classic_templates.templates
        class_templates = classic_waterbirds_templates.classes
    elif args.dataset == 'celeba':
        template = classic_templates.templates
        class_templates = classic_celeba_templates.classes
    else:
        raise NotImplementedError
    
    if args.text_embeddings:
        text_dict = {}
    
    with torch.no_grad():
        zeroshot_weights = []
        for class_keywords in class_templates:
            texts = [template[0].format(class_keywords)]
            texts = clip.tokenize(texts).cuda()

            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            
            if args.text_embeddings:
                text_dict[template[0].format(class_keywords)] = class_embedding.clone().detach().cpu().numpy().tolist()
            
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()

    if args.text_embeddings:
        # save the dictionary to a json file
        emb_dir = os.path.join(args.data_dir, args.embedding_dir, args.dataset)
        if not os.path.exists(emb_dir):
            os.makedirs(emb_dir, exist_ok=True)
        file_path = os.path.join(emb_dir, 'text_embedding.json')
        
        import json
        with open(file_path, 'w') as f:
            json.dump(text_dict, f)
            print("save text emb")
        # print(text_dict)
        del text_dict

    if args.image_embeddings:
        image_dict = {}

    if args.split != 'all':
        if args.dataset == 'waterbirds':
            data_dir = os.path.join(args.data_dir, 'waterbirds', 'waterbird_complete95_forest2water2') # NOTE: 수정
            dataset = Waterbirds(data_dir=data_dir, split=args.split, transform=transform)
        elif args.dataset == 'celeba':
            data_dir = os.path.join(args.data_dir, 'celeba')
            dataset = CelebA(data_dir=data_dir, split=args.split, transform=transform)
        else:
            raise NotImplementedError
    
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, num_workers=4, drop_last=False)
        temperature = 0.02  # redundant parameter
    
        preds_minor, preds, targets_minor = [], [], []
        with torch.no_grad():
            for (image, (target, target_g, target_s, target_split), file_name) in tqdm(dataloader):

                image = image.cuda()
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                logits = image_features @ zeroshot_weights / temperature

                probs = logits.softmax(dim=-1).cpu()
                conf, pred = torch.max(probs, dim=1)

                ## GROUP label prediciton 결과
                if args.dataset == 'waterbirds':
                    # minor group if
                    # (target, target_s) == (0, 1): landbird on water background
                    # (target, target_s) == (1, 0): waterbird on land background
                    is_minor_pred = (((target == 0) & (pred == 1)) | ((target == 1) & (pred == 0))).long()
                    is_minor = (((target == 0) & (target_s == 1)) | ((target == 1) & (target_s == 0))).long()
                if args.dataset == 'celeba':
                    # minor group if
                    # (target, target_s) == (1, 1): blond man
                    is_minor_pred = ((target == 1) & (pred == 1)).long()
                    is_minor = ((target == 1) & (target_s == 1)).long()
                
                preds_minor.append(is_minor_pred)
                preds.append(pred)
                targets_minor.append(is_minor)

                if args.image_embeddings or args.predictions:               
                    for index, y, group, cofounder, split, img_emb, pred in zip(file_name, target, target_g, target_s, target_split, image_features, pred):
                        if args.dataset == 'waterbirds':
                            idx = "/".join(index.split("/")[-2:])
                            key_list = ['y', 'place', 'group', 'split', 'image_embedding', 'y_pred']
                            image_dict[idx] = dict.fromkeys(key_list)
                            image_dict[idx]['y'] = str(y.cpu().numpy())
                            image_dict[idx]['group'] = str(group.cpu().numpy())
                            image_dict[idx]['place'] = str(cofounder.cpu().numpy())
                            image_dict[idx]['split'] = str(split.cpu().numpy())
                            image_dict[idx]['image_embedding'] = img_emb.clone().detach().cpu().numpy().tolist()
                            image_dict[idx]['y_pred'] =str(pred.clone().detach().cpu().numpy())

                        if args.dataset == 'celeba':
                            idx = str(os.path.split(index)[-1])
                            key_list = ['blond', 'male', 'group', 'split', 'image_embedding', 'y_pred']
                            image_dict[idx] = dict.fromkeys(key_list)
                            image_dict[idx]['blond'] = str(y.cpu().numpy())
                            image_dict[idx]['group'] = str(group.cpu().numpy())
                            image_dict[idx]['male'] = str(cofounder.cpu().numpy())
                            image_dict[idx]['split'] = str(split.cpu().numpy())
                            image_dict[idx]['image_embedding'] = img_emb.clone().detach().cpu().numpy().tolist()
                            image_dict[idx]['y_pred'] =str(pred.clone().detach().cpu().numpy())

        preds_minor, preds, targets_minor = torch.cat(preds_minor), torch.cat(preds), torch.cat(targets_minor)
        print(classification_report(targets_minor, preds_minor))


    else:
        for split in ['train', 'val', 'test']:
            if args.dataset == 'waterbirds':
                data_dir = os.path.join(args.data_dir, 'waterbirds', 'waterbird_complete95_forest2water2') # NOTE: 수정
                dataset = Waterbirds(data_dir=data_dir, split=split, transform=transform)
            elif args.dataset == 'celeba':
                data_dir = os.path.join(args.data_dir, 'celeba')
                dataset = CelebA(data_dir=data_dir, split=split, transform=transform)
            else:
                raise NotImplementedError
        
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=4, drop_last=False)
            temperature = 0.02  # redundant parameter

            preds_minor, preds, targets_minor = [], [], []
            with torch.no_grad():
                for (image, (target, target_g, target_s, target_split), file_name) in tqdm(dataloader):

                    image = image.cuda()
                    image_features = model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    logits = image_features @ zeroshot_weights / temperature

                    probs = logits.softmax(dim=-1).cpu()
                    conf, pred = torch.max(probs, dim=1)

                    ## GROUP label prediciton 결과
                    if args.dataset == 'waterbirds':
                        # minor group if
                        # (target, target_s) == (0, 1): landbird on water background
                        # (target, target_s) == (1, 0): waterbird on land background
                        is_minor_pred = (((target == 0) & (pred == 1)) | ((target == 1) & (pred == 0))).long()
                        is_minor = (((target == 0) & (target_s == 1)) | ((target == 1) & (target_s == 0))).long()
                    if args.dataset == 'celeba':
                        # minor group if
                        # (target, target_s) == (1, 1): blond man
                        is_minor_pred = ((target == 1) & (pred == 1)).long()
                        is_minor = ((target == 1) & (target_s == 1)).long()
                    
                    preds_minor.append(is_minor_pred)
                    preds.append(pred)
                    targets_minor.append(is_minor)

                    if args.image_embeddings or args.predictions:               
                        for index, y, group, cofounder, split, img_emb, pred in zip(file_name, target, target_g, target_s, target_split, image_features, pred):
                            if args.dataset == 'waterbirds':
                                idx = "/".join(index.split("/")[-2:])
                                key_list = ['y', 'place', 'group', 'split', 'image_embedding', 'y_pred']
                                image_dict[idx] = dict.fromkeys(key_list)
                                image_dict[idx]['y'] = str(y.cpu().numpy())
                                image_dict[idx]['group'] = str(group.cpu().numpy())
                                image_dict[idx]['place'] = str(cofounder.cpu().numpy())
                                image_dict[idx]['split'] = str(split.cpu().numpy())
                                image_dict[idx]['image_embedding'] = img_emb.clone().detach().cpu().numpy().tolist()
                                image_dict[idx]['y_pred'] =str(pred.clone().detach().cpu().numpy())

                            if args.dataset == 'celeba':
                                idx = str(os.path.split(index)[-1])
                                key_list = ['blond', 'male', 'group', 'split', 'image_embedding', 'y_pred']
                                image_dict[idx] = dict.fromkeys(key_list)
                                image_dict[idx]['blond'] = str(y.cpu().numpy())
                                image_dict[idx]['group'] = str(group.cpu().numpy())
                                image_dict[idx]['male'] = str(cofounder.cpu().numpy())
                                image_dict[idx]['split'] = str(split.cpu().numpy())
                                image_dict[idx]['image_embedding'] = img_emb.clone().detach().cpu().numpy().tolist()
                                image_dict[idx]['y_pred'] =str(pred.clone().detach().cpu().numpy())

            preds_minor, preds, targets_minor = torch.cat(preds_minor), torch.cat(preds), torch.cat(targets_minor)
            print(classification_report(targets_minor, preds_minor))


    if args.image_embeddings or args.predictions:
        import json
        emb_dir = os.path.join(args.data_dir, args.embedding_dir, args.dataset, args.backbone)
        if not os.path.exists(emb_dir):
            os.makedirs(emb_dir, exist_ok=True)
        file_path = os.path.join(emb_dir, 'embedding_prediction.json')
        with open(file_path, 'w') as f:
            json.dump(image_dict, f)
            print(f"dataset size: {len(image_dict)}")
            print("save img and pred")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--dataset', default='celeba', choices=['celeba', 'waterbirds'])
    parser.add_argument('--split', default='celeba', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--backbone', default='RN50', choices=['RN50', 'RN101', 'RN50x4', 'ViT-B/32'])
    parser.add_argument('--embedding_dir', default='./embeddings')
    parser.add_argument('--prediction_dir', default='./predictions')
    parser.add_argument('--text_embeddings', default=True, action='store_true')
    parser.add_argument('--image_embeddings', default=True, action='store_true')
    parser.add_argument('--predictions',  default=True, action='store_true')

    args = parser.parse_args()
    main(args)
