
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import Pl_model_wrapper
from models import models_with_args
import numpy as np
import random
import time
import pickle
from torch_geometric.loader import DataLoader
from datasets.cnf_data import Problem, Sat_datamodule, solve_sat, InMemorySATDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import tqdm
import pickle
from models.decimation import get_labeled_embs_and_votes
from models.decimation import group_clusters
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



model_name = 'NeuroSAT'
checkpoint = 'temp/tb_logs/Final/version_0/checkpoints/epoch=32-step=891.ckpt' #TODO args
num_iters = 80
model_class = models_with_args[model_name]
loss_fn =  nn.BCEWithLogitsLoss()
model = Pl_model_wrapper(model_class, 
                -1, 
                -1,
                loss_fn,
                return_embs=True)
model = model.load_from_checkpoint(checkpoint)
model.eval()

folder_name = 'temp/cnfs/selsam_3_40/test/' # TODO args

d = model_class['model_args']['d']
k = 1
# iterate of all problems in the folder, only the filenames with .dimacs extension are considered

def filter_files(folder_name,substr):
    filtered = []
    for file_name in tqdm.tqdm(os.listdir(folder_name)):
        if file_name.endswith(".dimacs") and substr in file_name:
            filtered.append(file_name)
    return filtered
filtered = filter_files(folder_name,"0040")

with torch.no_grad():
    labeled_embs, num_wrong_clusterings, num_wrong_polarities, y_s = get_labeled_embs_and_votes(filtered, d, k, model, num_iters, folder_name)

means_0 = []
means_1 = []
for pr in labeled_embs:
    means_0.append(np.mean(pr[1]['0'],axis=0))
    means_1.append(np.mean(pr[1]['1'],axis=0))
means_all = means_0+means_1
# KMeans clustering
clusterer = KMeans(n_clusters=2, random_state=10)
cluster_labels = clusterer.fit_predict(means_all)

groups = group_clusters(means_all, cluster_labels)
zero_centers = np.mean(groups['0'],axis=0)
ones_centers = np.mean(groups['1'],axis=0)
cluster_centers = {'0':zero_centers, '1':ones_centers}
# save cluster centers to pickle
with open('temp/cluster_centers.pkl', 'wb') as f: #TODO args
    pickle.dump(cluster_centers, f)