# Feature Debiasing with Regularized Adapter in Vision-Language Foundation Model

- Jinsu Park (GSAI, Postech)  
- Sangwoon Lim (GSAI, Postech)  
- Joonwon Jang (GSAI, Postech)  

## Requirements
We include a `requirements.txt` file for installing dependencies with `pip install -r requirements.txt`.  

List of (main) library:  
torch==1.12.1  
torchvision==0.13.1  
tqdm==4.65.0  
umap-learn==0.5.3  
scikit-learn==1.2.2  
scipy==1.10.1  
sklearn==0.0.post4  
easydict==1.10  
matplotlib==3.7.1  
numpy==1.23.5  
pandas==2.0.0  

## Download datasets and preprocessing

### 1. Download datasets
**Waterbirds**: Download the dataset from [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz), and move unzipped folder to `./data/waterbirds/waterbird_complete95_forest2water2`  

**CelebA**: Download the dataset from [here](https://www.kaggle.com/jessicali9530/celeba-dataset), and move unzipped files to `./data/celeba/[files]`.  


### 2. Extract embeddings using CLIP model.

```
# celeba 
python clip_inference.py --data_dir data --dataset celeba --embedding_dir embeddings_unnormalized --save --split all --backbone RN50

# waterbirds
python clip_inference.py --data_dir data --dataset waterbirds --embedding_dir embeddings_unnormalized --save --split all --backbone RN50
```

The extractred text/images embeddings would be saved in `data/embeddings_unnormalized/[dataset]`.


## Train and Evaluation

- Warnings: you should append the path of main folder using `sys.path.append(....)`, if library importing errors occur.

- Train(single): refer to [`run_final_main.sh`](https://github.com/Lainshower/debiasing-multi-modal/blob/main/run_final_main.sh)  
- Train(multiple(mean+-std); celeba): refer to [`run_multiple/run_final_main_iteration_ca.sh`](https://github.com/Lainshower/debiasing-multi-modal/blob/main/run_multiple/run_final_main_iteration_ca.sh)  
- Train(multiple(mean+-std); waterbirds): refer to [`run_multiple/run_final_main_iteration_wb.sh`](https://github.com/Lainshower/debiasing-multi-modal/blob/main/run_multiple/run_final_main_iteration_wb.sh)  

Note that best hyper-parameter is described in these run files. 

## Demo with corresponding results in reports.
### Table 2
> See the training demo in [`demo/demo_train.ipynb`](https://github.com/Lainshower/debiasing-multi-modal/blob/main/demo/demo_train.ipynb) in the case of **celeba** dataset. 

<img width="1172" alt="image" src="https://github.com/Lainshower/debiasing-multi-modal/assets/71121461/8ecc5b00-4309-4db8-a366-4f66be8fc75f">


- Full sweeped results can be found in [`demo/results_waterbirds.out`](https://github.com/Lainshower/debiasing-multi-modal/blob/main/demo/results_waterbirds.out) about **waterbirds** dataset
(best model was selected from these results. refer to [`demo/demo_final_performance_and_ablations.ipynb`]((https://github.com/Lainshower/debiasing-multi-modal/blob/main/demo/demo_final_performance_and_ablations.ipynb))).  
  


### Table 3
> See the results in table of ablation study, in [`demo/demo_final_performance_and_ablations.ipynb`](https://github.com/Lainshower/debiasing-multi-modal/blob/main/demo/demo_final_performance_and_ablations.ipynb).
<img width="1183" alt="image" src="https://github.com/Lainshower/debiasing-multi-modal/assets/71121461/a174cb3a-96b0-49fd-a198-5abc19114d96">


This also include the performance of our final model(*GCP-Seq-MA*).  


### Figures
>  See the visualization results in [`demo/demo_visualization.ipynb`](https://github.com/Lainshower/debiasing-multi-modal/blob/main/demo/demo_visualization.ipynb)
<img width="1175" alt="image" src="https://github.com/Lainshower/debiasing-multi-modal/assets/71121461/6303e1ab-7fd4-4648-bf5d-995e17f17ee3">
<img width="1186" alt="image" src="https://github.com/Lainshower/debiasing-multi-modal/assets/71121461/0a58e3e1-a438-4f51-87cd-1be073416374">


