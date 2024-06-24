import torch
import pytorch_lightning as pl
import numpy as np

class Pl_model_wrapper(pl.LightningModule):
    def __init__(self, 
                 model, 
                 lr, 
                 weight_decay,
                 loss_fn,
                 return_embs = False
                ):
        super(Pl_model_wrapper, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        model_class = model['model_class']
        model_args = model['model_args']
        if return_embs:
            model_args['return_embs'] = True
        self.model = model_class(**model_args)
        self.loss_fn = loss_fn

        rnd = torch.randn(16, requires_grad=False)
        self.true_vec = torch.nn.functional.normalize(rnd, p=2, dim=0)
        self.num_iters = 30

    def forward(self, batch):
        
        return self.model(batch,self.num_iters)

    def training_step(self, batch, batch_idx):
        
        otp = self(batch)      
        positive_lit_otp = [o[:o.shape[0]//2] for o in otp[-1]]
        products = torch.cat(positive_lit_otp, dim =0).squeeze(1)
        
        
        split_points = list((batch.num_literals//2).detach().cpu().numpy())
        closest_solutions = []
        for ix, split_point in enumerate(split_points):
            start_ix = sum(split_points[:ix])
            x = products[start_ix:split_point+start_ix]
            y = batch.sampled_solutions[start_ix:split_point+start_ix, :]
            minimal_ix = torch.argmin(torch.sum(((y.permute(1, 0) - x)**2), dim=1))
            closest_solutions.append(y[:, minimal_ix])
        tar = torch.cat(closest_solutions)

        #Výměna labelu podle blízkosti řešení nebo 1 řešení
        loss = self.loss_fn(products, batch.sat_assignment) #"tar" nebo "batch.sat_assignment", tar pro nejbližší sat_assignment pro jedno
        self.log('train_loss',loss,prog_bar=True, logger=True)
        
       
        return loss

    def validation_step(self, batch, batch_idx):
        
        otp = self(batch)

        positive_lit_otp = [o[:o.shape[0]//2] for o in otp[-1]]
        products = torch.cat(positive_lit_otp, dim =0).squeeze(1)
        loss = self.loss_fn(products, batch.sat_assignment)
        self.log('val_loss',loss,prog_bar=True, logger=True)

        infered_assignments = [torch.sign(stuff) for stuff in positive_lit_otp]
        gaps = []
        solved = 0

        for num in range(len(otp[0])):
            result_literals = []
            for ix, assignment in enumerate(list(infered_assignments[num].detach().cpu().numpy())):
                result_literals.append(int((ix+1) * assignment))

            sat_num = 0
            for c in batch.clauses[num]:
                for lit in c:
                    if lit in result_literals:
                        sat_num +=1
                        break
            gap = len(batch.clauses[num])-sat_num
            if gap == 0:
                solved+=1
            gaps.append(gap)

        acc = solved/len(otp[0]) 

        self.log('val_avg_gap', np.mean(gaps), prog_bar=True, logger=True)
        self.log('val acc', acc, prog_bar=True, logger=True)
                
        return loss

    def test_step(self, batch, batch_idx):
        #NEIMPLEMENTOVANO - nefunguje
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