# Arguments
### dataset : [waterbirds, celeba]
### epochs : 100 in waterbirds, 50 in celeba
### learning_rate : [1e-3, 1e-2, 1e-1, 1, 3, 10]
### batch_size : [128, 256, 512, 1024]
### tl_method : [linear_probing, adapter, contrastive_adpater]
### train_target : [class, spurious]

dataset="waterbirds" # [waterbirds, celeba]

tl_method="adapter" # [linear_probing, adapter, contrastive_adpater]
target="class" # [class, spurious]
non_target="spurious" # [spurious, class] 

python main_ebd_classifier.py \
--epochs 100 --learning_rate 1e-1 --batch_size 128 --dataset ${dataset} \
--text_embedding_dir /home/jinsu/workstation/project/debiasing-multi-modal/data/embeddings_unnormalized/${dataset}/clip_${target}.json \
--text_spurious_embedding_dir /home/jinsu/workstation/project/debiasing-multi-modal/data/embeddings_unnormalized/${dataset}/clip_${non_target}.json \
--image_embedding_dir /home/jinsu/workstation/project/debiasing-multi-modal/data/embeddings_unnormalized/${dataset}/RN50/clip.json \
--data_dir  /home/jinsu/workstation/project/debiasing-multi-modal/data/${dataset}/waterbird_complete95_forest2water2 \
--tl_method  ${tl_method} \
--train_target ${target}