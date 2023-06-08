# python clip_inference.py --data_dir data --dataset waterbirds --embedding_dir embeddings --embeddings --predictions --save_path minor_pred/waterbirds.pt

list="RN50" #  RN101 RN50x4 ViT-B/32"
 
for var in $list
do
  python clip_inference.py --data_dir data --dataset celeba --embedding_dir embeddings_unnormalized --save --split all --backbone $var
done