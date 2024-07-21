import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import Pl_model_wrapper
from models import models_with_args
import numpy as np
import random
import time
import pickle
from datasets.cnf_data import get_CNF_dataset, Sat_datamodule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

import os
print(os.getcwd())

def nonincremental_train(model,
                   dataset,
                   batch_size,
                   max_epochs,
                   grad_clip,
                   num_iters,
                   logger,
                   checkpoint,
                   device):

    model.num_iters = num_iters
    data = Sat_datamodule(dataset, batch_size)
    
    trainer = pl.Trainer(max_epochs=max_epochs, 
                         logger=logger,
                         accelerator=device, devices=1,
                         gradient_clip_val=grad_clip)
    trainer.fit(model, data)
    trainer.save_checkpoint(checkpoint)
    return model


seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# set hyperparameters
lr = 2e-3
weight_decay = 1e-10
model_name = 'NeuroSATRNN'
logger = TensorBoardLogger("temp/tb_logs", name="Final",)


checkpoint = "final_model_checkpoint.ckpt" #None #'lightning_logs/version_679/checkpoints/epoch=49-step=3950.ckpt'
data_path =  "temp/cnfs/selsam_3_40" #"temp/cnfs/3sat_100_400"
print(data_path)

incremental = True # T, F

batch_size = 128

gpus = []
grad_clip = 0.65
num_iters = 30
max_epochs = 200

dataset = get_CNF_dataset(data_path)
model_class = models_with_args[model_name]
loss_fn =  nn.MSELoss()
model = Pl_model_wrapper(model_class, 
                lr, 
                weight_decay,
                loss_fn,
                True)
print(model.device)
device = "cuda"

model = nonincremental_train(model,
                        dataset,
                        batch_size,
                        max_epochs,
                        grad_clip,
                        num_iters,
                        logger,
                        checkpoint,
                        device)