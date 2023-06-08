# Best parameter
## base batch size : 1024
## reg batch size : 4
## base lr : 0.1
## reg lr : base lr * 10.0
## Multiple Adapter : True
## Warm-up Reg : True 
## Balanced reg set : True

epochs=65
dataset="celeba"
data_dir="/home/jinsu/workstation/project/debiasing-multi-modal/data/celeba"
epochs_feature_learning=40 # LR 1e-1.. 이하일 때에나 충분히 Fitting이 이루어지긴 하나, 일단 이 방법론에도 Robust해야하는 게 맞으니까.
lr_decay_epochs='62,64'

tl_method="adapter_reg_seq_alter"

target="class" # [class, spurious]
non_target="spurious" # [spurious, class] 

bs_list="1024"
bsr_list="4,8,16" 
lr_list="1e-1"

CUDA_VISIBLE_DEVICES=1 python final_run_iteration_ca.py \
--epochs ${epochs} --epochs_feature_learning ${epochs_feature_learning} \
--lr_list ${lr_list} --bs_list ${bs_list} --bsr_list ${bsr_list} \
--dataset ${dataset} \
--text_embedding_dir /home/jinsu/workstation/project/debiasing-multi-modal/data/embeddings_unnormalized/${dataset}/clip_${target}.json \
--text_spurious_embedding_dir /home/jinsu/workstation/project/debiasing-multi-modal/data/embeddings_unnormalized/${dataset}/clip_${non_target}.json \
--text_group_embedding_dir /home/jinsu/workstation/project/debiasing-multi-modal/data/embeddings_unnormalized/${dataset}/clip_group.json \
--image_embedding_dir /home/jinsu/workstation/project/debiasing-multi-modal/data/embeddings_unnormalized/${dataset}/RN50/clip.json \
--data_dir  ${data_dir} \
--tl_method  ${tl_method} \
--train_target ${target} \
--warm_reg --lr_decay_rate 0.1 --lr_decay_epochs ${lr_decay_epochs} --save_results --add_adapter --balance_val --lr_multiple 10.0 \
--num_iter 2