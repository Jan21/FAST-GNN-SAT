import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import Pl_model_wrapper
from models import models_with_args
import numpy as np
import random
import time
import pickle
import tqdm
import os
from data.cnf_data import get_CNF_dataset, InMemorySATDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from models.decimation import process_one,\
                    try_assignments_per_formula,\
                    get_new_problems_from_decimation,\
                    get_assignments_from_embs_single,\
                    try_assignments_and_return_not_solved,\
                    get_assignments_from_embs

import argparse

def test(model,
         folder_name,
         num_iters,
         d,
         k,
         decimation,
         cluster_centers,
         dec_thresh=1.9):
    model.num_iters = num_iters
    model.eval()
    num_unsat = 0
    num_not_solved = 0
    num_solved_first = 0
    num_solved_second = 0
    num_total = 0
    for i,file_name in enumerate(tqdm.tqdm(os.listdir(folder_name))):
        #if i==250:
        #    break
        if file_name.endswith(".dimacs") or file_name.endswith(".cnf"):
            num_total += 1
            file_path = os.path.join(folder_name, file_name)
            n_vars, clauses = InMemorySATDataset.parse_dimacs(None, file_path)
            res = process_one(n_vars, clauses, d, k, model, num_iters)
            if not res:
                num_unsat += 1
                continue
            assignments = get_assignments_from_embs_single(res,cluster_centers)    
            assignments = {'assignments':assignments,'clauses':clauses}
            sat_assignments,unsat_assignments = try_assignments_per_formula(assignments)
            if len(sat_assignments) == 0 and decimation:
                to_be_decimated = {'assignments':unsat_assignments,'clauses':clauses}
                new_problems,was_decimated = get_new_problems_from_decimation(to_be_decimated,dec_thresh)
                if not was_decimated:
                    num_not_solved += 1
                    continue
                else:
                    for pr in new_problems:
                        if pr['num_vars']==0:
                            continue
                        num_vars,clauses = pr['num_vars'],pr['clauses']
                        res = process_one(num_vars, clauses, d, k, model, num_iters)
                        assignments = get_assignments_from_embs_single(res,cluster_centers)    
                        assignments = {'assignments':assignments,'clauses':clauses}
                        sat_assignments,unsat_assignments = try_assignments_per_formula(assignments) 
                        if len(sat_assignments) > 0:
                            num_solved_second += 1
                            break   
                    num_not_solved += 1                   
            elif len(sat_assignments) > 0:
                num_solved_first += 1
            else:
                num_not_solved += 1
    print('num_unsat',num_unsat)
    print('num_not_solved',num_not_solved)
    print('num_solved_first',num_solved_first)
    print('num_solved_second',num_solved_second)
    print('num_total',num_total)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run test')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--datapath', type=str, help='Path to SAT data directory')
    parser.add_argument('--ccpath', type=str, help='Path to cluster centers pickle')
    parser.add_argument('--k', type=int, help='Parameter k')
    parser.add_argument('--dec', type=int, help='Use decimation')
    parser.add_argument('--num_iters', type=int, help='Number of iterations')
    args = parser.parse_args()



    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    lr = 2e-5
    weight_decay = 1e-10
    model_name = 'NeuroSAT'

    checkpoint = args.checkpoint #TODO args   
    #checkpoint = 'lightning_logs/colab/epoch=4-step=175.ckpt'
    data_path = args.datapath
    #data_path = 'temp/cnfs/selsam_3_40/test/'
    #data_path = 'temp/cnfs/bv/'
    with open(args.ccpath, 'rb') as f:
        cluster_centers = pickle.load(f)
    num_iters = args.num_iters
    d = 16
    k = args.k 
    decimation = args.dec
    
    model_class = models_with_args[model_name]
    loss_fn =  nn.BCEWithLogitsLoss()
    model = Pl_model_wrapper(model_class, 
                    lr, 
                    weight_decay,
                    loss_fn)
    if checkpoint:
        model = model.load_from_checkpoint(checkpoint)
    with torch.no_grad():
        test(model, data_path, num_iters, d, k, decimation, cluster_centers)

