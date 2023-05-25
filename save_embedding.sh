# python clip_inference.py --data_dir data --dataset waterbirds --embedding_dir embeddings --embeddings --predictions --save_path minor_pred/waterbirds.pt


# 0511, jinsu
# python clip_inference_js.py --data_dir data --dataset waterbirds --embedding_dir embeddings --text_embeddings --image_embeddings --predictions --split all
# python clip_inference_js.py --data_dir data --dataset celeba --embedding_dir embeddings --text_embeddings --image_embeddings --predictions --split all

# 0512, joonwon
 
list="RN50" #  RN101 RN50x4 ViT-B/32"
 
for var in $list
do
  python clip_inference_including_group_with_unnorm.py --data_dir data --dataset celeba --embedding_dir embeddings_unnormalized --save --split all --backbone $var
done