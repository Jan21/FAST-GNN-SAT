import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data.dataset import files_exist, __repr__
from torch_sparse import SparseTensor
import pytorch_lightning as pl
from os import listdir
import os.path as osp
from collections import defaultdict
import PyMiniSolvers.minisolvers as minisolvers

__all__ = ['solve_sat']


def solve_sat(n_vars, iclauses):
    solver = minisolvers.MinisatSolver()

    for i in range(n_vars):
        solver.new_var(dvar=True) # dvar=True <- this var will be used as a decision var

    for iclause in iclauses:
        solver.add_clause(iclause)

    is_sat = solver.solve()
    stats = solver.get_stats() # dictionary of solver statistics

    return is_sat, stats

class Problem(Data):
    # a Problem is a bipartite graph

    def __init__(self, edge_index=None, x_l=None, x_c=None, y=None):
        # edge_index is a bipartite adjacency matrix between lits and clauses
        # x_l is the feature vector of the lits nodes
        # x_c is the feature vector of the clauses nodes
        # y is the label: 1 if sat, 0 if unsat

        super(Problem, self).__init__()
        # nodes features
        self.x_l = x_l
        self.x_c = x_c
        self.y = y

        self.num_literals = x_l.size(0) if x_l is not None else 0
        self.num_clauses = x_c.size(0) if x_c is not None else 0

        # edges
        self.edge_index = edge_index
        self.adj_t = SparseTensor(row = edge_index[1],
                                  col = edge_index[0],
                                  sparse_sizes = [self.num_clauses, self.num_literals]
                                 ) if edge_index is not None else 0

        # compute number of variables
        assert self.num_literals %2 == 0
        self.num_vars = self.num_literals // 2

        self.num_nodes = self.num_literals + self.num_clauses

    def __inc__(self, key, value,store):
        if key == 'edge_index':
            return torch.tensor([[self.x_l.size(0)], [self.x_c.size(0)]])
        else:
            return super().__inc__(key, value)


# create pytorch geometric dataset

# Remark: using a InMemoryDataset is faster than a standard Dataset

class InMemorySATDataset(InMemoryDataset):
    def __init__(self, root, d):
        # root: location of dataset
        # d: number of features for x_l and x_c

        self.root = root
        self.d = d

        # create initial feature vectors
        self.l_init = torch.normal(mean=0.0, std=1.0, size=(1,self.d)) # original
        self.c_init = torch.normal(mean=0.0, std=1.0, size=(1,self.d)) # original
        #self.l_init = torch.zeros(size=(1,self.d), dtype=torch.float32)
        #self.c_init = torch.zeros(size=(1,self.d), dtype=torch.float32)
        self.denom = torch.sqrt(torch.tensor(self.d, dtype=torch.float32)) # or float64???

        super(InMemorySATDataset, self).__init__(root=self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return self.root

    @property
    def raw_file_names(self):
        # get dimacs filenames
        sorted_file_names = sorted([f for f in listdir(self.root)
                                   if osp.isfile(osp.join(self.root,f))])
        return sorted_file_names

    @property
    def processed_file_names(self):
        return ['data.pt']


    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        for raw_path in self.raw_paths:
            n_vars, clauses = self.parse_dimacs(raw_path)
            # n_vars is the number of variables according to the dimacs file
            # clauses is a list of lists (=clauses) of numbers (=literals)

            y, _ = solve_sat(n_vars, clauses) # get problem label (sat/unsat)

            # create graph instance (Problem)
            p = self.create_problem(n_vars, clauses, y)

            data_list.append(p)
        #for i in range(10):
        #    with open(self.root+f"/processed{str(i)}.pkl",'wb') as f:
        #        pickle.dump(data_list[20000*i:20000*i+20000], f)
        #with open(self.root+"/processed2.pkl",'wb') as f:
        #    pickle.dump(data_list[100000:], f)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    # parse dimacs file
    def parse_dimacs(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        i = 0
        while lines[i].strip().split(" ")[0] == "c":
            # strip : remove spaces at the beginning and at the end of the string
            i += 1

        header = lines[i].strip().split(" ")
        assert(header[0] == 'p')
        n_vars = int(header[2])
        clauses = [[int(s) for s in line.strip().split(" ")[:-1]] for line in lines[i+1:]]
        return n_vars, clauses

    # create Problem instance (graph) from parsed dimacs
    def create_problem(self, n_vars, clauses, y):
        # d is the number of features of x_l and x_c

        n_lits = int(2 * n_vars)
        n_clauses = len(clauses)

        # create feature vectors for lits and clauses
        x_l = (torch.div(self.l_init, self.denom)).repeat(n_lits, 1)
        x_c = (torch.div(self.c_init, self.denom)).repeat(n_clauses, 1)

        # get graph edges from list of clauses
        edge_index = [[],[]]
        for i,clause in enumerate(clauses):
            # get idxs of lits in clause
            lits_indices = [self.from_lit_to_idx(l, n_vars) for l in clause]
            clauses_indices = len(clause) * [i]

            # add all edges connected to clause i to edge_index
            edge_index[0].extend(lits_indices)
            edge_index[1].extend(clauses_indices)

        # convert edge_index to tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        return Problem(edge_index, x_l, x_c, y)

    def from_lit_to_idx(self, lit, n_vars):
        # from a literal in range {1,...n_vars,-1,...,-n_vars} get the literal
        # index in {0,...,n_lits-1} = {0,...,2*n_vars-1}
        # if l is positive l <- l-1
        # if l in negative l <- n_vars-l-1
        assert(lit!=0)
        if lit > 0 :
            return lit - 1
        if lit < 0 :
            return n_vars - lit - 1

    def from_index_to_lit(self, idx, n_vars):
        # inverse of 'from_lit_to_idx', just in case
        if idx < n_vars:
            return idx+1
        else:
            return n_vars-idx-1


def get_CNF_dataset(dataset_folder):
    dimacs_dir_train = osp.join(dataset_folder,"train")
    dimacs_dir_val = osp.join(dataset_folder,"val")
    dimacs_dir_test = osp.join(dataset_folder,"test")
    d = 128
    dataset_train = InMemorySATDataset(dimacs_dir_train, d)
    dataset_val = InMemorySATDataset(dimacs_dir_val, d)
    dataset_test = InMemorySATDataset(dimacs_dir_test, d)
    return [dataset_train, dataset_val, dataset_test]


class Sat_datamodule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size):
        super(Sat_datamodule, self).__init__()
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

    def group_data(dataset):
        # create a dictionary of train sets based on the number of literals
        train_sets = defaultdict(list)
        for pr in dataset:
            num_lits = pr.x_l.shape[0]
            train_sets[num_lits].append(pr)
        return train_sets