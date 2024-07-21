import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.nn import TransformerConv
from torch_sparse import matmul
from torch_geometric.typing import Adj, Size

class LCMessages(MessagePassing):
    def __init__(self, d):
        # aggr set to None (maybe have to change it)
        super(LCMessages, self).__init__(aggr=None)
        self.d = d
        self.C_u = nn.LSTM(input_size=d,
                               hidden_size=d,
                               bias=True)

    def message_and_aggregate(self, adj_t, x_l):
        return matmul(adj_t, x_l) # n_clauses x d



    def forward(self, adj_t, x_l, x_c, x_c_h):
        msg = self.propagate(adj_t, x_l=x_l) # num_clauses x d
        _, (x_c, x_c_h) = self.C_u(msg.unsqueeze(0),
                                   (x_c.unsqueeze(0),#.detach(),
                                    x_c_h.unsqueeze(0))#.detach())
                                  )
        return x_c.squeeze(0), x_c_h.squeeze(0)


class CLMessages(MessagePassing):

    def __init__(self, d):
        super(CLMessages, self).__init__(aggr=None)
        self.d = d
        self.L_u = nn.LSTM(input_size=2*d,
                               hidden_size=d,
                               bias=True)

    def message_and_aggregate(self, adj, x_c):
        return matmul(adj, x_c) # n_clauses x d


    def flip(self, L, L_batch):
        # count nb of lits in each problem:
        # nondeterministic version
        lits_per_prob = torch.bincount(L_batch).detach()
        # deterministic version; x10 slower
        #batch_size = torch.max(L_batch)+1
        #lits_per_prob = torch.zeros(batch_size).to(device=L.device)
        #for n in range(batch_size):
        #    lits_per_prob[n] = torch.count_nonzero(L_batch == n)
        #lits_per_prob = lits_per_prob.int()
        # start position of each new problem in batch:
        start_probs = torch.roll(torch.cumsum(lits_per_prob, dim=0),1).tolist()
        start_probs[0] = 0
        lits_per_prob = lits_per_prob.tolist()
        # swap literals of all problems
        L_flipped = torch.empty(0,L.size(1)).to(L.device)

        for n_lits, start_pos in zip(lits_per_prob, start_probs):
            assert n_lits % 2 == 0, "The number of literals is not even."
            n_vars = n_lits // 2
            L_flipped = torch.cat((L_flipped,
                                   L[start_pos+n_vars:start_pos+2*n_vars],
                                   L[start_pos:start_pos+n_vars]),
                                  dim=0)
        assert L.size() == L_flipped.size(), \
               "L and L_flipped sizes are different"
        return L_flipped


    def forward(self, adj_t, x_c, x_l, x_l_h, x_l_batch):
        # updates the values of x_l and x_l_h using x_c and flip(x_l)
        # x_l and x_l_h are [num_lits x d]
        # x_c is [num_clauses x d]
        # adj_t is [n_clauses x n_lits]
        msg = self.propagate(adj_t.t(), x_c=x_c) # num_lits x d
        msg_concat = torch.cat((msg, self.flip(x_l, x_l_batch)), dim=1) # num_lits x 2*d
        _, (x_l, x_l_h) = self.L_u(msg_concat.unsqueeze(0),
                                   (x_l.unsqueeze(0),#.detach(),
                                    x_l_h.unsqueeze(0))#.detach())
                                  )
        return x_l.squeeze(0), x_l_h.squeeze(0)

class NeuroSAT(nn.Module):

    def __init__(self, d,
                 n_msg_layers=0,
                 n_vote_layers=0,
                 mlp_transfer_fn = 'relu',
                 final_reducer = 'mean',
                 lstm = 'standard',
                 return_embs = False,
                ):
        super(NeuroSAT, self).__init__()

        self.d = d
        self.return_embs = return_embs
        self.final_reducer = final_reducer
        self.init_ts = torch.ones(1)
        self.init_ts.requires_grad = False

        self.L_init = nn.Linear(1, d)
        self.C_init = nn.Linear(1, d)

        self.LC_msgs = LCMessages(d=d)
        self.CL_msgs = CLMessages(d=d)
        self.L_vote = nn.Linear(d, 1)

        self.true_vec_mult = torch.nn.Linear(d, 1, bias=False)

    def forward(self,data,num_iters):
        adj_t = data.adj_t
        n_lits, n_clauses = data.x_l.shape[0], data.x_c.shape[0]
        
        #initialize x_l and x_c
        init_ts = self.init_ts.to(data.x_l.device)
        x_l = torch.rand((n_lits,self.d),requires_grad=False).to(data.x_l.device)
        
        C_init = self.C_init(init_ts)
        x_c = C_init.repeat(n_clauses, 1)

        x_l_batch = data.x_l_batch
        # initialize lstm cell states
        x_l_h = torch.zeros(x_l.shape).to(data.x_l.device)
        x_c_h = torch.zeros(x_c.shape).to(data.x_l.device)

        for t in range(num_iters):
            x_c_, x_c_h = self.LC_msgs(adj_t, x_l, x_c, x_c_h)
            x_l, x_l_h = self.CL_msgs(adj_t, x_c_, x_l, x_l_h, x_l_batch)
            x_c = x_c_
        #return x_l
        x_l_vote = self.L_vote(x_l)

        if self.return_embs:
            # group by x_l_batch
            x_l_ = [x_l[x_l_batch==i] for i in range(data.x_l_batch.max()+1)]
            x_l_vote_ = [x_l_vote[x_l_batch==i] for i in range(data.x_l_batch.max()+1)]
            truth_assignment = [self.true_vec_mult(emb_mat) for emb_mat in x_l_]

            return x_l_,x_l_vote_,global_mean_pool(x_l_vote, x_l_batch), truth_assignment
        
        if self.final_reducer == 'mean':
            logits_average_vote = global_mean_pool(x_l_vote, x_l_batch)
        else:
            raise NotImplementedError

        return logits_average_vote



class LCMessagesRNN(MessagePassing):
    def __init__(self, d):
        # aggr set to None (maybe have to change it)
        super(LCMessagesRNN, self).__init__(aggr=None)
        self.d = d
        self.C_u = nn.RNN(input_size=d,
                               hidden_size=d,
                               bias=True)

    def message_and_aggregate(self, adj_t, x_l):
        return matmul(adj_t, x_l) # n_clauses x d



    def forward(self, adj_t, x_l, x_c):
        msg = self.propagate(adj_t, x_l=x_l) # num_clauses x d
        _, x_c = self.C_u(msg.unsqueeze(0),x_c.unsqueeze(0))
                                   #.detach(),
                                    #.detach()
                                  
        return x_c.squeeze(0)


class CLMessagesRNN(MessagePassing):

    def __init__(self, d):
        super(CLMessagesRNN, self).__init__(aggr=None)
        self.d = d
        self.L_u = nn.RNN(input_size=2*d,
                               hidden_size=d,
                               bias=True)

    def message_and_aggregate(self, adj, x_c):
        return matmul(adj, x_c) # n_clauses x d


    def flip(self, L, L_batch):
        # count nb of lits in each problem:
        # nondeterministic version
        lits_per_prob = torch.bincount(L_batch).detach()
        # deterministic version; x10 slower
        #batch_size = torch.max(L_batch)+1
        #lits_per_prob = torch.zeros(batch_size).to(device=L.device)
        #for n in range(batch_size):
        #    lits_per_prob[n] = torch.count_nonzero(L_batch == n)
        #lits_per_prob = lits_per_prob.int()
        # start position of each new problem in batch:
        start_probs = torch.roll(torch.cumsum(lits_per_prob, dim=0),1).tolist()
        start_probs[0] = 0
        lits_per_prob = lits_per_prob.tolist()
        # swap literals of all problems
        L_flipped = torch.empty(0,L.size(1)).to(L.device)

        for n_lits, start_pos in zip(lits_per_prob, start_probs):
            assert n_lits % 2 == 0, "The number of literals is not even."
            n_vars = n_lits // 2
            L_flipped = torch.cat((L_flipped,
                                   L[start_pos+n_vars:start_pos+2*n_vars],
                                   L[start_pos:start_pos+n_vars]),
                                  dim=0)
        assert L.size() == L_flipped.size(), \
               "L and L_flipped sizes are different"
        return L_flipped


    def forward(self, adj_t, x_c, x_l, x_l_batch):
        # updates the values of x_l and x_l_h using x_c and flip(x_l)
        # x_l and x_l_h are [num_lits x d]
        # x_c is [num_clauses x d]
        # adj_t is [n_clauses x n_lits]
        msg = self.propagate(adj_t.t(), x_c=x_c) # num_lits x d
        msg_concat = torch.cat((msg, self.flip(x_l, x_l_batch)), dim=1) # num_lits x 2*d
        _, x_l = self.L_u(msg_concat.unsqueeze(0), x_l.unsqueeze(0))
                                   #.detach(),
                                    #.detach())
                                  
        return x_l.squeeze(0)

class NeuroSATRNN(nn.Module):

    def __init__(self, d,
                 n_msg_layers=0,
                 n_vote_layers=0,
                 mlp_transfer_fn = 'relu',
                 final_reducer = 'mean',
                 lstm = 'standard',
                 return_embs = False,
                ):
        super(NeuroSATRNN, self).__init__()

        self.d = d
        self.return_embs = return_embs
        self.final_reducer = final_reducer
        self.init_ts = torch.ones(1)
        self.init_ts.requires_grad = False

        self.L_init = nn.Linear(1, d)
        self.C_init = nn.Linear(1, d)

        self.LC_msgs = LCMessagesRNN(d=d)
        self.CL_msgs = CLMessagesRNN(d=d)
        self.L_vote = nn.Linear(d, 1)

        self.true_vec_mult = torch.nn.Linear(d, 1, bias=False)

    def forward(self,data,num_iters):
        adj_t = data.adj_t
        n_lits, n_clauses = data.x_l.shape[0], data.x_c.shape[0]
        
        #initialize x_l and x_c
        #init_ts = self.init_ts.to(data.x_l.device)
        x_l = torch.rand((n_lits,self.d),requires_grad=False).to(data.x_l.device)
        x_l = x_l / torch.norm(x_l, dim=1, keepdim=True)
        
        x_c = torch.rand((n_clauses,self.d),requires_grad=False).to(data.x_l.device)
        x_c = x_c / torch.norm(x_c, dim=1, keepdim=True)
        
        
        
        #C_init = self.C_init(init_ts)
        #x_c = C_init.repeat(n_clauses, 1)

        x_l_batch = data.x_l_batch
        # initialize lstm cell states
        #x_l_h = torch.zeros(x_l.shape).to(data.x_l.device)
        #x_c_h = torch.zeros(x_c.shape).to(data.x_l.device)

        truth_assignments = []

        x_l_ = [x_l[x_l_batch==i] for i in range(data.x_l_batch.max()+1)]
        truth_assignment = [self.true_vec_mult(emb_mat) for emb_mat in x_l_]
        #print(truth_assignment)
        truth_assignments.append(truth_assignment)

        clause_embs = []
        cl_embs = [x_c[data.x_c_batch==i] for i in range(data.x_c_batch.max()+1)]
        clause_embs.append(cl_embs)
        
        for t in range(num_iters):
            x_c_= self.LC_msgs(adj_t, x_l, x_c)
            x_l = self.CL_msgs(adj_t, x_c_, x_l, x_l_batch)
            x_l = x_l / torch.norm(x_l, dim=1, keepdim=True)
            x_c = x_c_
            x_c = x_c / torch.norm(x_c, dim=1, keepdim=True)

            #NEW PART to extract intermediate steps
            x_l_ = [x_l[x_l_batch==i] for i in range(data.x_l_batch.max()+1)]
            truth_assignment = [self.true_vec_mult(emb_mat) for emb_mat in x_l_]
            truth_assignments.append(truth_assignment)
        
            cl_embs = [x_c[data.x_c_batch==i] for i in range(data.x_c_batch.max()+1)]
            clause_embs.append(cl_embs)

        #return x_l
        x_l_vote = self.L_vote(x_l)

        
        # group by x_l_batch
        #x_l_ = [x_l[x_l_batch==i] for i in range(data.x_l_batch.max()+1)]
        x_l_vote_ = [x_l_vote[x_l_batch==i] for i in range(data.x_l_batch.max()+1)]
        #truth_assignment = [self.true_vec_mult(emb_mat) for emb_mat in x_l_]

        
        return {"final_lits_votes" : x_l_vote_ , "final_lits_mats" : x_l_, "vote_mean_pool" : global_mean_pool(x_l_vote, x_l_batch), "final_truth_assignment" : truth_assignments[-1],
                "each_step_truth_assignments" : truth_assignments, "clause_embs_all_steps" : clause_embs, "initial_truth_assignment" : truth_assignments[0]}