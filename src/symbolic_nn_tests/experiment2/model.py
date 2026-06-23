from functools import lru_cache

import torch
from torch import nn
from torchmetrics.regression import R2Score

from symbolic_nn_tests.models import PubChemModel as Model


# This is just a quick, lazy way to ensure all models are trained on the same dataset
@lru_cache(maxsize=1)
def get_singleton_dataset():
    from symbolic_nn_tests.dataloader import create_dataset
    from symbolic_nn_tests.experiment2.dataset import collate, pubchem

    return create_dataset(
        dataset=pubchem,
        collate_fn=collate,
        batch_size=256,
        shuffle_train=True,
        num_workers=0,
    )


def unpacking_smooth_l1_loss(out, y):
    _, y_pred = out
    return nn.functional.smooth_l1_loss(y_pred, y)


class UnpackingR2Score(R2Score):
    def update(self, preds, target):
        if isinstance(preds, tuple):
            _, y_pred = preds
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]
        else:
            y_pred = preds
        super().update(y_pred.view(-1), target.view(-1))


def main(
    train_loss=unpacking_smooth_l1_loss,
    val_loss=unpacking_smooth_l1_loss,
    test_loss=unpacking_smooth_l1_loss,
    logger=None,
    trainer_callbacks=None,
    semantic_trainer=False,
    **kwargs,
):
    import lightning as L

    if semantic_trainer:
        from .train import TrainingWrapper
    else:
        from symbolic_nn_tests.train import TrainingWrapper

    if logger is None:
        from lightning.pytorch.loggers import TensorBoardLogger

        logger = TensorBoardLogger(save_dir=".", name="logs/ffnn")

    train, val, test = get_singleton_dataset()
    lmodel = TrainingWrapper(
        Model(),
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=test_loss,
        accuracy=UnpackingR2Score(),
    )
    lmodel.configure_optimizers(optimizer=torch.optim.NAdam, **kwargs)
    trainer = L.Trainer(max_epochs=5, logger=logger, callbacks=trainer_callbacks)
    trainer.fit(model=lmodel, train_dataloaders=train, val_dataloaders=val)
    trainer.test(dataloaders=test)


if __name__ == "__main__":
    main()
