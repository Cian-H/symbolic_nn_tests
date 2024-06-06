from functools import lru_cache
import torch
from torch import nn


model = nn.Sequential(
    nn.Flatten(1, -1),
    nn.Linear(784, 10),
    nn.Softmax(dim=1),
)


def collate(batch):
    x, y = zip(*batch)
    x = [i[0] for i in x]
    y = [torch.tensor(i) for i in y]
    x = torch.stack(x).to("cuda")
    y = torch.tensor(y).to("cuda")
    return x, y


# This is just a quick, lazy way to ensure all models are trained on the same dataset
@lru_cache(maxsize=1)
def get_singleton_dataset():
    from torchvision.datasets import QMNIST

    from symbolic_nn_tests.dataloader import create_dataset

    return create_dataset(
        dataset=QMNIST, collate_fn=collate, batch_size=128, shuffle=True
    )


def oh_vs_cat_cross_entropy(y_bin, y_cat):
    return nn.functional.cross_entropy(
        y_bin,
        nn.functional.one_hot(y_cat),
    )


def main(
    train_loss=oh_vs_cat_cross_entropy,
    val_loss=oh_vs_cat_cross_entropy,
    test_loss=oh_vs_cat_cross_entropy,
    logger=None,
    **kwargs,
):
    import lightning as L

    from symbolic_nn_tests.train import TrainingWrapper

    if logger is None:
        from lightning.pytorch.loggers import TensorBoardLogger

        logger = TensorBoardLogger(save_dir=".", name="logs/ffnn")

    train, val, test = get_singleton_dataset()
    lmodel = TrainingWrapper(
        model, train_loss=train_loss, val_loss=val_loss, test_loss=val_loss
    )
    lmodel.configure_optimizers(**kwargs)
    trainer = L.Trainer(max_epochs=20, logger=logger)
    trainer.fit(model=lmodel, train_dataloaders=train, val_dataloaders=val)
    trainer.test(dataloaders=test)


if __name__ == "__main__":
    main()
