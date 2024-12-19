import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import Pl_model_wrapper
from models import models_with_args
import numpy as np
import random
import time
import pickle
from data.cnf_data import get_CNF_dataset, Sat_datamodule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

import argparse


def nonincremental_train(model,
                   dataset,
                   batch_size,
                   max_epochs,
                   grad_clip,
                   num_iters,
                   logger,
                   checkpoint):

    model.num_iters = num_iters
    data = Sat_datamodule(dataset, batch_size)
    trainer = pl.Trainer(max_epochs=max_epochs, 
                         logger=logger,
                         accelerator="gpu", devices=1,
                         gradient_clip_val=grad_clip)
    trainer.fit(model, data)
    trainer.save_checkpoint(checkpoint)
    return model

# incremental training
def incremental_train(model,
                    dataset,
                    batch_size,
                    max_epochs,
                    grad_clip,
                    num_steps,
                    logger,
                    checkpoint): 
    train_dataset = Sat_datamodule.group_data(dataset[0])
    val_dataset = Sat_datamodule.group_data(dataset[1])
    train_sizes = sorted(train_dataset.keys())
    val_sizes = sorted(val_dataset.keys())
    print('val sizes: ',val_sizes)
    # train on each size incrementally
    threshs = np.linspace(0.65, 0.85, len(train_sizes))
    num_last_k_sizes = 5 # the last k sizes to train on
    results_per_size = []
    for i,s in enumerate(train_sizes):
        start = time.time()
        # the first training size will be 10 literals
        # after that we skip every other size except the last before the final one
        if i%2==0 and s < 78 and s > 10 or s < 10:
            continue
        thresh = threshs[i]
        print('current thresh: ',threshs[i])
        trainer = pl.Trainer(max_epochs=max_epochs, 
                            #gpus=gpus,
                            accelerator="gpu", devices=1,
                            logger=logger,
                            gradient_clip_val=grad_clip,
                            callbacks=[EarlyStopping(monitor="val_acc", 
                                                    patience=max_epochs,
                                                    check_on_train_epoch_end = False,
                                                    stopping_threshold = thresh,
                                                    mode = 'max')])
        train_dataset_ = train_dataset[s]
        num_probs_of_size_s_ = len(train_dataset_) #//min(i,num_last_k_sizes)
        
        for s_ in train_sizes:
            if s_ < (s-2*num_last_k_sizes):
                continue
            if s_ >= s:
                break
            train_dataset_ += train_dataset[s_][:num_probs_of_size_s_]
        
        model.num_iters=min(num_steps,max(6,s//2))
        print(len(train_dataset_))
        print(f"Training on size {s}")
        dataset_ = [train_dataset_,val_dataset[s],dataset[2]]
        data = Sat_datamodule(dataset_, batch_size)
        trainer.fit(model, data)
        end = time.time()
        t_diff = end-start
        res = trainer.validate(model, data.test_dataloader())
        results_per_size.append((res,t_diff))
    trainer.save_checkpoint(checkpoint)
    return results_per_size,model
                             
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--datapath', type=str, help='Path to SAT data directory')
    parser.add_argument('--incremental', default=True, type=int, help='Whether to run incremental training')
    args = parser.parse_args()

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # set hyperparameters
    lr = 2e-3
    weight_decay = 1e-10
    model_name = 'NeuroSAT'
    logger = TensorBoardLogger("temp/tb_logs", name="Final",)

    checkpoint = "final_model_checkpoint.ckpt" #args.checkpoint #None #'lightning_logs/version_679/checkpoints/epoch=49-step=3950.ckpt'
    data_path =  args.datapath #"/home/jan/projects/CIIRC/backdoors/FAST-GNN-SAT/temp/cnfs/selsam_3_40"
    
    incremental = args.incremental
    batch_size = 128
    gpus = [0]
    grad_clip = 0.65
    num_iters = 30
    max_epochs = 200
    
    # create dataset and model
    dataset = get_CNF_dataset(data_path)
    model_class = models_with_args[model_name]
    loss_fn =  nn.BCEWithLogitsLoss()
    model = Pl_model_wrapper(model_class, 
                    lr, 
                    weight_decay,
                    loss_fn)
    #if checkpoint:
    #    model = model.load_from_checkpoint(checkpoint)

    if incremental:
        results,model = incremental_train(model,
                                           dataset,
                                           batch_size,
                                           max_epochs,
                                           grad_clip,
                                           num_iters,
                                           logger,
                                           checkpoint)   
        # save results to pickle file
        with open('temp/incremental_results.pkl','wb') as f:
            pickle.dump(results, f)                              
    else:
        model = nonincremental_train(model,
                            dataset,
                           batch_size,
                            max_epochs,
                            grad_clip,
                            num_iters,
                            logger,
                            checkpoint)
     
