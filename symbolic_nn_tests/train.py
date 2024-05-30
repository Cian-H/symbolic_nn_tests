from torch import nn, optim
import lightning as L


class TrainingWrapper(L.LightningModule):
    def __init__(self, model, loss_func=nn.functional.mse_loss, accuracy=None):
        super().__init__()
        self.model = model
        self.loss_func = loss_func
        self.accuracy = accuracy

    def _forward_step(self, batch, batch_idx, label=""):
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss_func(y_pred, y)
        self.log(f"{label}{'_' if label else ''}loss", loss)
        if self.accuracy is not None:
            acc = self.accuracy(y_pred, y)
            self.log(f"{label}{'_' if label else ''}acc", acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self._forward_step(batch, batch_idx, label="train")

    def validation_step(self, batch, batch_idx):
        self._forward_step(batch, batch_idx, label="val")

    def test_step(self, batch, batch_idx):
        self._forward_step(batch, batch_idx, label="test")

    def configure_optimizers(self, optimizer=optim.SGD, **kwargs):
        _optimizer = optimizer(self.parameters(), **kwargs)
        return _optimizer
