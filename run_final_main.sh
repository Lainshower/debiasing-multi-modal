# Arguments
epochs=100
dataset="waterbirds"
data_dir="/home/jinsu/workstation/project/debiasing-multi-modal/data/waterbirds/waterbird_complete95_forest2water2"

epochs_feature_learning=40
lr_decay_epochs='90,95'

tl_method="adapter_reg_seq_alter" # [linear_probing, adapter, adapter_reg, adapter_reg_seq, adapter_reg_seq_alter]

target="class" # [class, spurious]
non_target="spurious" # [spurious, class] 

# Best hyper-parameter in waterbirds.
bs=1024
bsr=256
lr=1.0
lrr=1.0

CUDA_VISIBLE_DEVICES=3 python main_ebd_classifier_group_seq_alter_balval_resampled.py \
--epochs ${epochs} --learning_rate ${lr} --batch_size ${bs} \
--epochs_feature_learning ${epochs_feature_learning} --learning_rate_reg ${lrr} --batch_size_reg ${bsr} \
--dataset ${dataset} \
--text_embedding_dir /home/jinsu/workstation/project/debiasing-multi-modal/data/embeddings_unnormalized/${dataset}/clip_${target}.json \
--text_spurious_embedding_dir /home/jinsu/workstation/project/debiasing-multi-modal/data/embeddings_unnormalized/${dataset}/clip_${non_target}.json \
--text_group_embedding_dir /home/jinsu/workstation/project/debiasing-multi-modal/data/embeddings_unnormalized/${dataset}/clip_group.json \
--image_embedding_dir /home/jinsu/workstation/project/debiasing-multi-modal/data/embeddings_unnormalized/${dataset}/RN50/clip.json \
--data_dir  ${data_dir} \
--tl_method  ${tl_method} \
--train_target ${target} \
--watch_batch_results --print_freq 1 --save_results \
--warm_reg --lr_decay_rate 0.1 --lr_decay_epochs ${lr_decay_epochs} --add_adapter --random_seed 42