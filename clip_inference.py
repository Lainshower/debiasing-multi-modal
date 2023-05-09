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
from data.waterbirds import Waterbirds
import classic_templates
import classic_celeba_templates
import classic_waterbirds_templates

def main(args):
    model, preprocess = clip.load('RN50', device='cuda', jit=False)  # RN50, RN101, RN50x4, ViT-B/32

    crop = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])
    transform = transforms.Compose([crop, preprocess])

    if args.dataset == 'waterbirds':
        data_dir = os.path.join(args.data_dir, 'waterbirds')
        train_dataset = Waterbirds(data_dir=data_dir, split='train', transform=transform)
        template = classic_templates.templates
        class_templates = classic_waterbirds_templates.classes
    elif args.dataset == 'celeba':
        data_dir = os.path.join(args.data_dir, 'celeba')
        train_dataset = CelebA(data_dir=data_dir, split='train', transform=transform)
        template = classic_templates.templates
        class_templates = classic_celeba_templates.classes
    else:
        raise NotImplementedError

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, num_workers=4, drop_last=False)
    temperature = 0.02  # redundant parameter
    
    if args.embeddings:
        text_dict, image_dict = {}, {}
    
    with torch.no_grad():
        zeroshot_weights = []
        for class_keywords in class_templates:
            texts = [template[0].format(class_keywords)]
            texts = clip.tokenize(texts).cuda()

            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            
            if args.embeddings:
                text_dict[template[0].format(class_keywords)] = class_embedding.clone().detach().cpu().numpy()
            
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()

    if args.embeddings:
        # save the dictionary to a numpy file
        emb_dir = os.path.join(args.data_dir, args.embedding_dir, args.dataset)
        if not os.path.exists(emb_dir):
            os.makedirs(emb_dir, exist_ok=True)

        # save the dictionary to a numpy file
        file_path = os.path.join(emb_dir, 'text_embedding.npy')
        np.save(file_path, text_dict)
        print(text_dict)
        print("save")

    if args.predictions:
        prediction_dict ={}

    preds_minor, preds, targets_minor = [], [], []
    with torch.no_grad():
        for (image, (target, target_g, target_s), file_name) in tqdm(train_dataloader):
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
            if args.embeddings or args.predictions:               
                for image, prediction, index in zip(image_features, pred, file_name):
                    if args.embeddings:
                        image_dict[os.path.split(index)[-1]] = image.clone().detach().cpu().numpy()
                    if args.predictions:
                        prediction_dict[os.path.split(index)[-1]] = str(prediction.clone().detach().cpu().numpy())
                # prediction이랑 image embedding file path 만들어주기
    
    if args.embeddings:
        # save the dictionary to a numpy file[]
        emb_dir = os.path.join(args.data_dir, args.embedding_dir, args.dataset, args.backbone)
        if not os.path.exists(emb_dir):
            os.makedirs(emb_dir, exist_ok=True)
        # save the dictionary to a numpy file
        file_path = os.path.join(emb_dir, 'image_embedding.npy')
        np.save(file_path, image_dict)
        print(image_dict)
        print("save img")

    if args.predictions:
        pred_dir = os.path.join(args.data_dir, args.prediction_dir, args.dataset,args.backbone)
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir, exist_ok=True)
        file_path = os.path.join(pred_dir, 'prediction.csv')
        
        import csv
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            # create a CSV writer object
            writer = csv.writer(file)
            writer.writerow(['image_id', 'prediction'])
            # write the data rows
            for key, value in prediction_dict.items():
                writer.writerow([key, value])
        
        # save the dictionary to a numpy file
        print(prediction_dict)
        print("save pred")

    preds_minor, preds, targets_minor = torch.cat(preds_minor), torch.cat(preds), torch.cat(targets_minor)

    print(classification_report(targets_minor, preds_minor))

    # Save minor group prediction results
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(preds, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--dataset', default='celeba', choices=['celeba', 'waterbirds'])
    parser.add_argument('--backbone', default='RN50', choices=['RN50', 'RN101', 'RN50x4', 'ViT-B/32'])
    parser.add_argument('--embedding_dir', default='./embeddings')
    parser.add_argument('--prediction_dir', default='./predictions')
    parser.add_argument('--embeddings', default=True, action='store_true')
    parser.add_argument('--predictions',  default=True, action='store_true')
    parser.add_argument('--save_path', default='./minor_pred/celeba.pt')

    args = parser.parse_args()
    main(args)
