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
wandb==0.14.2
tensorboard==2.12.2
tensorboard-data-server==0.7.0
tensorboard-plugin-wit==1.8.1
threadpoolctl==3.1.0

## Datasets and code 

**Waterbirds**: Download the dataset from [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz), and move unzipped folder to `./data/waterbirds/waterbird_complete95_forest2water2`

**CelebA**: Download the dataset from [here](https://www.kaggle.com/jessicali9530/celeba-dataset), and move unzipped files to `./data/celeba/[files]`.

## Train and Evaluation

Train(single): refer to `run_final_main.sh`
Train(multiple(mean+-std); celeba): refer to `run_final_main_iteration_ca.sh`
Train(multiple(mean+-std); waterbirds): refer to `run_final_main_iteration_wb.sh`  

## Demo with corresponding results in reports.
### Table 1

See the **Jupyter Notebook** Demo in `demo_train.ipynb` in celeba. 

Full sweeped results can be found in `results_waterbirds.out` about waterbirds dataset, and `results_celeba_out` about celeba datasets (best model was selected from these results. refer to `demo_final_performance_and_ablations.ipynb`).
  


### Table 3

See the results in table of ablation study, in `demo_final_performance_and_ablations.ipynb`.
This include the performance of our final model(*GCP-Seq-MA*).


### Figure 1, 3, 4

See the visualization results in `demo_visualization.ipynb`
