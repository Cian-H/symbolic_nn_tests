from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger
import torch
import wandb


def get_wandb_logger(loggers: list) -> WandbLogger:
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            break
    return logger


class ConfusionMatrixCallback(Callback):
    def __init__(self, class_names=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_names = class_names

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.state.stage != "sanity_check":
            y_pred = torch.concat(pl_module.epoch_step_preds)
            y_pred = torch.argmax(y_pred, axis=1)
            y = torch.concat(tuple(map(lambda xy: xy[1], trainer.val_dataloaders)))
            logger = get_wandb_logger(trainer.loggers)
            logger.experiment.log(
                {
                    f"confusion_matrix_epoch_{trainer.current_epoch}": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y.numpy(),
                        preds=y_pred.numpy(),
                        class_names=self.class_names,
                        title=f"confusion_matrix_epoch_{trainer.current_epoch}",
                    )
                }
            )
