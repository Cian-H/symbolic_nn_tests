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


class TrainingWrapper(L.LightningModule):
    def __init__(self, model, loss_func=nn.functional.cross_entropy):
        super().__init__()
        self.model = model
        self.loss_func = loss_func

    def _forward_step(self, batch, batch_idx, label=""):
        x, y = collate_batch(batch)
        y_pred = self.model(x)
        batch_size = x.shape[0]
        one_hot_y = nn.functional.one_hot(y).type(torch.float64)
        loss = self.loss_func(y_pred, one_hot_y)
        acc = torch.sum(y_pred.argmax(dim=1) == y) / batch_size
        self.log(f"{label}{'_' if label else ''}loss", loss, batch_size=batch_size)
        self.log(f"{label}{'_' if label else ''}acc", acc, batch_size=batch_size)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._forward_step(batch, batch_idx, label="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._forward_step(batch, batch_idx, label="val")

    def configure_optimizers(self, optimizer=optim.Adam, *args, **kwargs):
        _optimizer = optimizer(self.parameters(), *args, **kwargs)
        return _optimizer
