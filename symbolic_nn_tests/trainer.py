import torch
from torch import nn, optim
import lightning as L


def collate_batch(batch):
    x, y = zip(*batch)
    x = [i[0] for i in x]
    y = [torch.tensor(i) for i in y]
    x = torch.stack(x).to("cuda")
    y = torch.tensor(y).to("cuda")
    return x, y


class Trainer(L.LightningModule):
    def __init__(self, model, loss_func=nn.functional.cross_entropy):
        super().__init__()
        self.model = model
        self.loss_func = loss_func

    def training_step(self, batch, batch_idx):
        x, y = collate_batch(batch)
        y_pred = self.model(x)
        loss = self.loss_func(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = collate_batch(batch)
        y_pred = self.model(x)
        loss = self.loss_func(y_pred, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self, optimizer=optim.Adam, *args, **kwargs):
        _optimizer = optimizer(self.parameters(), *args, **kwargs)
        return _optimizer
