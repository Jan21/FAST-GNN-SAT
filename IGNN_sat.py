import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import DataLoader
from models import *
import numpy as np
import random
import time
from datasets.selsam import get_CNF_dataset
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from collections import defaultdict

# pytorch lightning model
class IGNN_sat(pl.LightningModule):
    def __init__(self, 
                 model, 
                 lr, 
                 weight_decay,
                 loss_fn,
                ):
        super(IGNN_sat, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        model_class = model['model_class']
        model_args = model['model_args']
        self.model = model_class(**model_args)
        self.loss_fn = loss_fn

    def forward(self, batch):
        return self.model(batch,self.num_iters)

    def training_step(self, batch, batch_idx):
        y = batch.y
        y_hat = self(batch)
        loss = self.loss_fn(y_hat.squeeze(), y.type_as(y_hat))
        self.log('train_loss',loss,prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch.y
        y_hat = self(batch)
        pred_binary = torch.sigmoid(y_hat.squeeze()) >= 0.5
        num_correct = (pred_binary == y.type_as(y_hat)).sum().item()
        num_total = len(y)
        acc = num_correct / num_total
        loss = self.loss_fn(y_hat.squeeze(), y.type_as(y_hat))
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        y = batch.y
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss)
        pred =  y_hat.max(dim=1)[1]
        acc = pred.eq(y).sum().item() / y.shape[0]
        self.log('test_acc', acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

# pytorch lightning dataloader
class IGNN_sat_data(pl.LightningDataModule):
    def __init__(self, dataset, batch_size):
        super(IGNN_sat_data, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.follow_batch = ['x_l','x_c']

    def setup(self, stage=None):
        self.train_dataset = self.dataset[0]
        self.val_dataset = self.dataset[1]
        self.test_dataset = self.dataset[2]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,  follow_batch=self.follow_batch, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, follow_batch=self.follow_batch, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,  follow_batch=self.follow_batch, batch_size=self.batch_size)

# pytorch lightning trainer
def train_IGNN_sat(model,
                   dataset,
                   batch_size,
                   max_epochs,
                   gpus,
                   grad_clip,
                   num_iters,
                   logger):

    model.num_iters = num_iters
    data = IGNN_sat_data(dataset, batch_size)
    trainer = pl.Trainer(max_epochs=max_epochs, 
                         logger=logger,
                         gradient_clip_val=grad_clip)
    trainer.fit(model, data)
    return model

def group_data(dataset):

    # create a dictionary of train sets based on the number of literals
    train_sets = defaultdict(list)
    for pr in dataset:
        num_lits = pr.x_l.shape[0]
        train_sets[num_lits].append(pr)
    return train_sets



# incremental training
def incremental_train_IGNN_sat(model,
                                dataset,
                                batch_size,
                                max_epochs,
                                gpus,
                                grad_clip,
                                num_steps,
                                logger,
                                additive_incremental=True): 
    train_dataset = group_data(dataset[0])
    val_dataset = group_data(dataset[1])

    train_sizes = sorted(train_dataset.keys())
    val_sizes = sorted(val_dataset.keys())
    print('val sizes: ',val_sizes)
    # train on each size incrementally
    threshs = np.linspace(0.65, 0.85, len(train_sizes))
    results_per_size = []
    for i,s in enumerate(train_sizes):
        start = time.time()
        #if i%2==0 and s > 20:
        #    continue
        if s < 10:
            continue
        thresh = threshs[i]
        print('current thresh: ',threshs[i])
        trainer = pl.Trainer(max_epochs=max_epochs, 
                            #gpus=gpus,
                            logger=logger,
                            gradient_clip_val=grad_clip,
                            callbacks=[EarlyStopping(monitor="val_acc", 
                                                    patience=max_epochs,
                                                    check_on_train_epoch_end = False,
                                                    stopping_threshold = thresh,
                                                    mode = 'max')])
        if additive_incremental:
            num_last_k_sizes = 5
            train_dataset_ = train_dataset[s]
            num_probs_of_size_s_ = len(train_dataset_) #//min(i,num_last_k_sizes)
            for s_ in train_sizes:
                if s_ < (s-2*num_last_k_sizes):
                  continue
                if s_ >= s:
                    break
                train_dataset_ += train_dataset[s_][:num_probs_of_size_s_]
        else:
            train_dataset_ = train_dataset[s]
        model.num_iters=min(num_steps,max(6,s//2))
        print(len(train_dataset_))
        print(f"Training on size {s}")
        dataset_ = [train_dataset_,val_dataset[s],val_dataset[s]]
        data = IGNN_sat_data(dataset_, batch_size)
        trainer.fit(model, data)
        end = time.time()
        t_diff = end-start
        res = trainer.validate(model, dataset[2])
        results_per_size.append((res,t_diff))
    return results_per_size
                             
if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # set hyperparameters
    lr = 2e-3
    weight_decay = 1e-10
    model_name = 'NeuroSAT'
    logger = TensorBoardLogger("temp/tb_logs", name="Final",)

    checkpoint = None #'lightning_logs/version_679/checkpoints/epoch=49-step=3950.ckpt'
    #checkpoint = 'lightning_logs/colab/epoch=4-step=175.ckpt'
    data_path = 'temp/cnfs/selsam_3_40'
    
    incremental = True
    batch_size = 128
    gpus = [0]
    grad_clip = 0.65
    num_iters = 30
    max_epochs = 200
    
    # create dataset and model
    dataset = get_CNF_dataset(data_path)
    model_class = models_with_args[model_name]
    loss_fn =  nn.BCEWithLogitsLoss()
    model = IGNN_sat(model_class, 
                    lr, 
                    weight_decay,
                    loss_fn)
    if checkpoint:
        model = model.load_from_checkpoint(checkpoint)

    if incremental:
        model = incremental_train_IGNN_sat(model,
                                           dataset,
                                           batch_size,
                                           max_epochs,
                                           gpus,
                                           grad_clip,
                                           num_iters,
                                           logger)                                  
    else:
        model = train_IGNN_sat(model,
                            dataset,
                            batch_size,
                            max_epochs,
                            gpus,
                            grad_clip,
                            num_iters,
                            logger)
