from functools import lru_cache
from torch import nn


model = nn.Sequential(
    nn.Flatten(1, -1),
    nn.Linear(784, 10),
    nn.Softmax(dim=1),
)


# This is just a quick, lazy way to ensure all models are trained on the same dataset
@lru_cache(maxsize=1)
def get_singleton_dataset():
    from torchvision.datasets import QMNIST

    from symbolic_nn_tests.dataloader import get_dataset

    return get_dataset(dataset=QMNIST)


def main(loss_func=nn.functional.cross_entropy, logger=None, **kwargs):
    import lightning as L

    from symbolic_nn_tests.train import TrainingWrapper

    if logger is None:
        from lightning.pytorch.loggers import TensorBoardLogger

        logger = TensorBoardLogger(save_dir=".", name="logs/ffnn")

    train, val, test = get_singleton_dataset()
    lmodel = TrainingWrapper(model, loss_func=loss_func)
    lmodel.configure_optimizers(**kwargs)
    trainer = L.Trainer(max_epochs=20, logger=logger)
    trainer.fit(model=lmodel, train_dataloaders=train, val_dataloaders=val)
    trainer.test(dataloaders=test)


if __name__ == "__main__":
    main()
