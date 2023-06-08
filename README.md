# debiasing-multi-modal


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

## Datasets and code 

### Download datasets
**Waterbirds**: Download the dataset from [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz), and move unzipped folder to `./data/waterbirds/waterbird_complete95_forest2water2`  

**CelebA**: Download the dataset from [here](https://www.kaggle.com/jessicali9530/celeba-dataset), and move unzipped files to `./data/celeba/[files]`.  


### Extract embeddings using CLIP model.

```
# celeba 
python clip_inference.py --data_dir data --dataset celeba --embedding_dir embeddings_unnormalized --save --split all --backbone RN50

# waterbirds
python clip_inference.py --data_dir data --dataset waterbirds --embedding_dir embeddings_unnormalized --save --split all --backbone RN50

```

The extractred text/images embeddings would be saved in `data/embeddings_unnormalized/[dataset]`  


## Train and Evaluation

- Warnings: you should append the path of main folder using `sys.path.append(....)`

Train(single): refer to `demo/run_final_main.sh`  
Train(multiple(mean+-std); celeba): refer to `demo/run_final_main_iteration_ca.sh`  
Train(multiple(mean+-std); waterbirds): refer to `demo/run_final_main_iteration_wb.sh`    

Note that best hyper-parameter is described in these run files. 

## Demo with corresponding results in reports.
### Table 1

See the **Jupyter Notebook** Demo in `demo/demo_train.ipynb` in celeba. 

Full sweeped results can be found in `demo/results_waterbirds.out` about waterbirds dataset, and `demo/results_celeba_out` about celeba datasets (best model was selected from these results. refer to `demo/demo_final_performance_and_ablations.ipynb`).  
  


### Table 3

See the results in table of ablation study, in `demo/demo_final_performance_and_ablations.ipynb`.
This include the performance of our final model(*GCP-Seq-MA*).  


### Figure 1, 3, 4

See the visualization results in `demo/demo_visualization.ipynb`
