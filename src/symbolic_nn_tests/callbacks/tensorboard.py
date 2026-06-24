import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from sklearn.metrics import confusion_matrix


class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")


class TensorBoardConfusionMatrixCallback(Callback):
    def __init__(self, class_names=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_names = class_names

    def log_confusion_matrix(self, trainer, pl_module, dataloaders, stage):
        if trainer.state.stage == "sanity_check":
            return

        if not pl_module.epoch_step_preds:
            return

        y_pred = torch.concat(pl_module.epoch_step_preds)

        # Unpack tuple if necessary
        if isinstance(y_pred, tuple):
            y_pred = y_pred[1]
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]

        # Only compute confusion matrix for classification tasks
        if y_pred.ndim == 1 or y_pred.shape[1] == 1:
            return

        y_pred = torch.argmax(y_pred, dim=1)

        try:
            # Try extracting y from test dataloader
            if isinstance(dataloaders, list):
                y = torch.concat(tuple(xy[1] for xy in dataloaders[0]))
            else:
                y = torch.concat(tuple(xy[1] for xy in dataloaders))
        except (TypeError, ValueError):
            return

        cm = confusion_matrix(y.numpy(), y_pred.numpy())
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(cm, cmap="Blues")
        fig.colorbar(cax)

        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), va="center", ha="center")

        if self.class_names:
            ax.set_xticks(range(len(self.class_names)))
            ax.set_yticks(range(len(self.class_names)))
            ax.set_xticklabels(self.class_names, rotation=45)
            ax.set_yticklabels(self.class_names)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix ({stage})", pad=20)
        plt.tight_layout()

        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_figure(
                    f"Confusion_Matrix/{stage}", fig, global_step=trainer.global_step
                )

        plt.close(fig)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.val_dataloaders:
            self.log_confusion_matrix(trainer, pl_module, trainer.val_dataloaders, "Validation")

    def on_test_epoch_end(self, trainer, pl_module):
        if trainer.test_dataloaders:
            self.log_confusion_matrix(trainer, pl_module, trainer.test_dataloaders, "Test")
