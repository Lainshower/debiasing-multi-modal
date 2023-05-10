import json

file_path ='./data/embeddings/waterbirds/RN50/embedding_prediction.json'
with open(file_path, 'r') as f:
    my_dict = json.load(f)

print(len(my_dict))
