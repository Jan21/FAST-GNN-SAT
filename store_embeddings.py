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
import tqdm

if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model_name = 'NeuroSAT'
    checkpoint = 'temp/tb_logs/Final/version_0/checkpoints/epoch=32-step=891.ckpt'
    data_path = 'temp/cnfs/selsam_3_40' 
    num_iters = 70
    dataset = get_CNF_dataset(data_path)
    model_class = models_with_args[model_name]
    print(model_class)
    loss_fn =  nn.BCEWithLogitsLoss()
    model = Pl_model_wrapper(model_class, 
                    0, 
                    0,
                    loss_fn,
                    return_embs=True)
    model = model.load_from_checkpoint(checkpoint)
    model.model.return_embs = True
    train_dataset = Sat_datamodule.group_data(dataset[0])[80]
    test_dataset = dataset[2]
    dataset_ = [train_dataset,dataset[2],dataset[2]]
    data = Sat_datamodule(dataset_, 1)
    data.setup()
    model.eval()
    train_embs = []
    test_embs = []
    for ex in tqdm.tqdm(data.train_dataloader()):
        train_embs.append({'embs':model.model(ex,num_iters)[0].detach().numpy(),'y':ex.y.detach().numpy()})   
    for ex in tqdm.tqdm(data.test_dataloader()):
        test_embs.append({'embs':model.model(ex,num_iters)[0].detach().numpy(),'y':ex.y.detach().numpy()})
    embs = {'train':train_embs,'test':test_embs}
    # save embs to pickle file
    with open('temp/embs_150.pkl','wb') as f:
        pickle.dump(embs, f)