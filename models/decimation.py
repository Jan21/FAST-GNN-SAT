
import PyMiniSolvers.minisolvers as minisolvers
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances

def create_problem(n_vars, clauses, d,k=1):
        # d is the number of features of x_l and x_c
        y, _ = solve_sat(n_vars, clauses) # get problem label (sat/unsat)
        n_lits = int(2 * n_vars)
        n_clauses = len(clauses)
        # create initial feature vectors
        l_init = torch.normal(mean=0.0, std=1.0, size=(1,d)) # original
        c_init = torch.normal(mean=0.0, std=1.0, size=(1,d)) # original
        denom = torch.sqrt(torch.tensor(d, dtype=torch.float32)) # or float64???
        # create feature vectors for lits and clauses
        x_l = (torch.div(l_init, denom)).repeat(n_lits, 1)
        x_c = (torch.div(c_init, denom)).repeat(n_clauses, 1)
        # get graph edges from list of clauses
        edge_index = [[],[]]
        for i,clause in enumerate(clauses):
            # get idxs of lits in clause
            lits_indices = [InMemorySATDataset.from_lit_to_idx(None,l, n_vars) for l in clause]
            clauses_indices = len(clause) * [i]

            # add all edges connected to clause i to edge_index
            edge_index[0].extend(lits_indices)
            edge_index[1].extend(clauses_indices)

        # convert edge_index to tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        problem = Problem(edge_index, x_l, x_c, y)
        # wraph it to a dataloader so that we can use the same code as in training
        dl =  DataLoader([problem for _ in range(k)],  follow_batch=['x_l','x_c'], batch_size=k)
        # get the example..quite ugly TODO: fix
        for batch in dl:
            return (batch,y)

def group_clusters(embedding, cluster_labels):
    cluster_0 = []
    cluster_1 = []
    for i, label in enumerate(cluster_labels):
        if label == 0:
            cluster_0.append(embedding[i])
        else:
            cluster_1.append(embedding[i])
    return {"0":cluster_0, "1":cluster_1}

def get_clustering_score(embedding):
    clusterer = KMeans(n_clusters=2, random_state=10)
    cluster_labels = clusterer.fit_predict(embedding)
    silhouette_avg = silhouette_score(embedding, cluster_labels)
    return silhouette_avg, cluster_labels

def check_sat_assignment(positive, clauses):
    for cl in clauses:
        clause_sat = False
        for lit in cl:
            if lit < 0:
                sign = 0
            else:
                sign = 1
            var_ix = np.abs(lit)-1
            bool_val = positive[var_ix]
            if bool_val == sign:
                clause_sat = True
                break
        if not clause_sat:
            return False
    return True 

def check_sat(clauses, positive, negative):
    if check_sat_assignment(positive, clauses):
        return (True,list(positive)+list(negative))
    if check_sat_assignment(negative, clauses):
        return (True,list(negative)+list(positive))
    else:
        return (False,None)

def process_one(n_vars, clauses, d, k, model, num_iters):
    model.model.return_embs = True
    p,y = create_problem(n_vars, clauses, d, k)
    p.cuda()
    model.cuda()
    if y == False:
        return False
    embs,votes,avg_votes = model.model(p,num_iters)
    pred_binary = torch.sigmoid(avg_votes.squeeze()) >= 0.5
    majority = torch.mode(pred_binary).values
    # take indices of the majority class
    res = {'embs': [emb.detach().cpu().numpy() for i,emb in enumerate(embs) ],#if i in majority_idx],
                    'votes': [vote.detach().cpu().numpy() for i,vote in enumerate(votes)], #if i in majority_idx],
                    'y_hat':majority.detach().cpu().numpy(),
                    'y':y,
                    'clauses':clauses}
    return res

def get_processed_data(filtered, d, k, model, num_iters, folder_name):
    data = []
    for file_name in tqdm.tqdm(filtered):
            n_vars, clauses = InMemorySATDataset.parse_dimacs(None,folder_name+file_name)
            res = process_one(n_vars, clauses, d, k, model, num_iters)
            res['file_name'] = file_name
            data.append(res)
    return data

def get_labeled_embs_and_votes(filtered, 
                               d, 
                               k, 
                               model, 
                               num_iters, 
                               folder_name,
                               processed_data=None):
    y_s = []
    labeled_embs = []
    num_wrong_clusterings = []
    num_wrong_polarities = []
    if processed_data is None:
        processed_data = get_processed_data(filtered, d, k, model, num_iters, folder_name)
    for data in tqdm.tqdm(processed_data):
            embs = data['embs'][0]
            clauses = data['clauses']
            y = data['y']
            sat_thresh = 0.6160 # estimated in the classif_by_clustering.ipynb
            score, cluster_labels = get_clustering_score(embs)
            if score < sat_thresh:
                continue
            num_vars = len(cluster_labels)//2
            positive = cluster_labels[:num_vars] 
            negative = cluster_labels[num_vars:] 
            # for each variable the sum of two polarities should 1
            polarities_check = sum(positive+negative)==num_vars
            if polarities_check:
                is_sat, assignment = check_sat(clauses, positive, negative)
                if is_sat:
                    labeled_groups = group_clusters(embs, assignment)
                    labeled_embs.append(("SAT",labeled_groups,clauses))
                else:
                    num_wrong_clusterings.append(y)
            else:
                num_wrong_polarities.append(y)
    return labeled_embs, num_wrong_clusterings, num_wrong_polarities, y_s

def get_unique_assignments(assignments):
    unique_assignments = []
    for assignment in assignments:
        if assignment not in unique_assignments:
            unique_assignments.append(assignment)
    return unique_assignments

def get_assignments_from_embs_single(dat,cluster_centers):
    embs = dat['embs']
    assignments_per_formula = []
    for emb in embs:
        assignment = []
        for lit in emb[:len(emb)//2]:
            center_0 = cluster_centers['0']
            center_1 = cluster_centers['1']
            # get euclidean distance to cluster centers
            dist_0 = np.linalg.norm(lit-center_0)
            dist_1 = np.linalg.norm(lit-center_1)
            if dist_0 < dist_1:
                assignment.append((0,dist_0))
            else:
                assignment.append((1,dist_1))
        assignments_per_formula.append(assignment)
    return assignments_per_formula

def get_assignments_from_embs(processed_data,cluster_centers):
    assignments = []
    for dat in processed_data:  
            y = dat['y']
            y_hat = dat['y_hat']
            assignments_per_formula = get_assignments_from_embs_single(dat,cluster_centers)
            assignments.append({'assignments':assignments_per_formula,
                                'clauses':dat['clauses'],
                                'y':y,
                                'y_hat':y_hat,})
    return assignments

def try_assignments_per_formula(assignments):
        sat_assignments_formula = []
        unsat_assignments_formula = []
        for a in assignments['assignments']:
            a,dists = [list(t) for t in zip(*a)]
            is_sat = check_sat_assignment(a, assignments['clauses'])
            if is_sat:
                sat_assignments_formula.append(a)
            else:
                unsat_assignments_formula.append((a,dists))
        return sat_assignments_formula, unsat_assignments_formula
        
def try_assignments_and_return_not_solved(assignments):
    sat_assignments = []
    not_solved = []
    for formula in assignments:
        clauses = formula['clauses']
        y_hat = formula['y_hat']
        y = formula['y']
        sat_assignments_formula, unsat_assignments_formula = try_assignments_per_formula(formula)
        if len(sat_assignments_formula)>0:
            sat_assignments.append({'assignments':sat_assignments_formula,'clauses':clauses})
        elif y==True:
            not_solved.append({'assignments':unsat_assignments_formula,'clauses':clauses})
    #print("num sat solved: ",len(sat_assignments))
    #print("num sat not solved: ",len(not_solved))
    return not_solved,sat_assignments

def remove_unused_vars(clauses,assignment):
    all_lits = []
    for cl in clauses:
        all_lits.extend(cl)
    all_vars = [np.abs(lit)-1 for lit in all_lits]
    unique_vars = set(all_vars)
    fixed = set(assignment.keys())
    unused_vars = unique_vars-fixed
    renamed_vars = {}
    num_vars = len(unused_vars)
    for i, var in enumerate(list(unused_vars)):
        renamed_vars[var] = i+1
    new_clauses = []
    for cl in clauses:
        new_cl = []
        for lit in cl:
            if lit < 0:
                sign = -1
            else:
                sign = 1
            var_ix = np.abs(lit)-1
            if var_ix in unused_vars:
                new_cl.append(sign*renamed_vars[var_ix])
        if len(new_cl)==0:
            continue
        new_clauses.append(new_cl)
    return new_clauses, num_vars

def simplify_sat(clauses,assignment):
    simplified_clauses = []
    for cl in clauses:
        clause_sat = False
        for lit in cl:
            if lit < 0:
                sign = 0
            else:
                sign = 1
            var_ix = np.abs(lit)-1
            if var_ix not in assignment.keys():
                continue
            bool_val = assignment[var_ix]
            if bool_val == sign:
                clause_sat = True
                break
        if not clause_sat:
            simplified_clauses.append(cl)
    return simplified_clauses

def decimate(dists,bools,clauses,dec_thresh):
    # get indices of literals that have dist < 1.5
    decimated = {}
    for i, dist in enumerate(dists):
        if dist < dec_thresh:
            decimated[i]=bools[i]
    if decimated == {}:
        return False, None
    new_clauses = simplify_sat(clauses,decimated)
    new_clauses,num_vars = remove_unused_vars(new_clauses,decimated)
    return new_clauses, num_vars

def solve_sat(n_vars, iclauses):
    solver = minisolvers.MinisatSolver()

    for i in range(n_vars):
        solver.new_var(dvar=True) # dvar=True <- this var will be used as a decision var

    for iclause in iclauses:
        solver.add_clause(iclause)

    is_sat = solver.solve()
    stats = solver.get_stats() # dictionary of solver statistics
    return is_sat, stats

def get_new_problems_from_decimation(pr,dec_thresh):
    clauses = pr['clauses']
    assignments = pr['assignments']
    was_decimated = False
    new_problems_per_formula = []
    for assignment in assignments:
        dists = assignment[1]
        bools = assignment[0]
        new_clauses, num_vars = decimate(dists,bools,clauses,dec_thresh)
        if new_clauses == False:
            continue
        else:
            was_decimated = True
        is_sat,stats = solve_sat(num_vars,new_clauses)
        if is_sat:
            new_problems_per_formula.append({'clauses':new_clauses,'num_vars':num_vars})
    return new_problems_per_formula, was_decimated