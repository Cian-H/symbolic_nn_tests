from torch import nn, optim
import lightning as L


class TrainingWrapper(L.LightningModule):
    def __init__(
        self,
        model,
        train_loss=nn.functional.mse_loss,
        val_loss=nn.functional.mse_loss,
        test_loss=nn.functional.mse_loss,
        accuracy=None,
    ):
        super().__init__()
        self.model = model
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.test_loss = val_loss
        self.accuracy = accuracy
        self.epoch_step_preds = []

    def _forward_step(self, batch, batch_idx, loss_func, label=""):
        x, y = batch
        y_pred = self.model(x)
        loss = loss_func(y_pred, y)
        # Add tracking of y_pred for each step in RAM (for more advanced plots)
        if batch_idx == 0:
            self.epoch_step_preds = []
        self.epoch_step_preds.append(y_pred.cpu())
        # Add enhanced logging for more granularity
        self.log(f"{label}{'_' if label else ''}loss", loss)
        if self.accuracy is not None:
            acc = self.accuracy(y_pred, y)
            self.log(f"{label}{'_' if label else ''}acc", acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self._forward_step(batch, batch_idx, self.train_loss, label="train")

    def validation_step(self, batch, batch_idx):
        self._forward_step(batch, batch_idx, self.val_loss, label="val")

    def test_step(self, batch, batch_idx):
        self._forward_step(batch, batch_idx, self.test_loss, label="test")

    def configure_optimizers(self, optimizer=optim.SGD, **kwargs):
        _optimizer = optimizer(self.parameters(), **kwargs)
        return _optimizer
