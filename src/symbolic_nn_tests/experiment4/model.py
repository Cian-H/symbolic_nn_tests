from functools import lru_cache

import torch
from torchmetrics import Accuracy

from symbolic_nn_tests.models import QMNISTModel as Model


def collate(batch):
    x, y = zip(*batch, strict=True)
    x = [i[0] for i in x]
    y = [torch.tensor(i) for i in y]
    x = torch.stack(x)
    y = torch.tensor(y)
    return x, y


@lru_cache(maxsize=1)
def get_singleton_dataset():
    from torchvision.datasets import QMNIST

    from symbolic_nn_tests.dataloader import create_dataset

    return create_dataset(
        dataset=QMNIST,
        collate_fn=collate,
        batch_size=256,
        shuffle_train=True,
        num_workers=0,
    )


def main(
    train_loss,
    val_loss,
    test_loss,
    accuracy=None,
    logger=None,
    trainer_callbacks=None,
    **kwargs,
):
    import lightning as L

    from symbolic_nn_tests.train import TrainingWrapper

    if accuracy is None:
        accuracy = Accuracy(task="multiclass", num_classes=10)

    if logger is None:
        from lightning.pytorch.loggers import TensorBoardLogger

        logger = TensorBoardLogger(save_dir=".", name="logs/ffnn")

    train, val, test = get_singleton_dataset()
    model = Model()
    lmodel = TrainingWrapper(
        model,
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=test_loss,
        accuracy=accuracy,
    )
    lmodel.configure_optimizers(**kwargs)
    trainer = L.Trainer(max_epochs=20, logger=logger, callbacks=trainer_callbacks)
    trainer.fit(model=lmodel, train_dataloaders=train, val_dataloaders=val)
    trainer.test(dataloaders=test)

    pass
