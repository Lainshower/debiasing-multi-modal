# Best parameter
## base batch size : 1024
## reg batch size : 256
## base lr : 1.0
## reg lr : base lr * 1.0
## Multiple Adapter : True
## Warm-up Reg : True 
## Balanced reg set : False (already near-balanced in waterbirds)

epochs=100
dataset="waterbirds"
data_dir="/home/jinsu/workstation/project/debiasing-multi-modal/data/waterbirds/waterbird_complete95_forest2water2"
epochs_feature_learning=40
lr_decay_epochs='75,90'

tl_method="adapter_reg_seq_alter"


target="class" # [class, spurious]
non_target="spurious" # [spurious, class] 

for bs in 512 1024; do
for bsr in 64 128 256 512; do
for lr in 1 10;do
CUDA_VISIBLE_DEVICES=0 python final_main_iteration_wb.py \
--epochs ${epochs} --learning_rate ${lr} --batch_size ${bs} \
--epochs_feature_learning ${epochs_feature_learning} --learning_rate_reg ${lr} --batch_size_reg ${bsr} \
--dataset ${dataset} \
--text_embedding_dir /home/jinsu/workstation/project/debiasing-multi-modal/data/embeddings_unnormalized/${dataset}/clip_${target}.json \
--text_spurious_embedding_dir /home/jinsu/workstation/project/debiasing-multi-modal/data/embeddings_unnormalized/${dataset}/clip_${non_target}.json \
--text_group_embedding_dir /home/jinsu/workstation/project/debiasing-multi-modal/data/embeddings_unnormalized/${dataset}/clip_group.json \
--image_embedding_dir /home/jinsu/workstation/project/debiasing-multi-modal/data/embeddings_unnormalized/${dataset}/RN50/clip.json \
--data_dir  ${data_dir} \
--tl_method  ${tl_method} \
--train_target ${target} \
--warm_reg --lr_decay_rate 0.1 --lr_decay_epochs ${lr_decay_epochs} --add_adapter
done
done
done