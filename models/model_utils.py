import torch
import pytorch_lightning as pl
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