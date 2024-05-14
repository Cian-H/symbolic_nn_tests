from torch import nn


model = nn.Sequential(
    nn.Flatten(1, -1),
    nn.Linear(784, 10),
    nn.Softmax(dim=1),
)


def main(loss_func=nn.functional.cross_entropy, logger=None):
    from torchvision.datasets import QMNIST
    import lightning as L

    from .dataloader import get_dataset
    from .train import TrainingWrapper

    if logger is None:
        from lightning.pytorch.loggers import TensorBoardLogger

        logger = TensorBoardLogger(save_dir=".", name="logs/ffnn")

    train, val, test = get_dataset(dataset=QMNIST)
    lmodel = TrainingWrapper(model)
    trainer = L.Trainer(max_epochs=5, logger=logger)
    trainer.fit(model=lmodel, train_dataloaders=train, val_dataloaders=val)


if __name__ == "__main__":
    main()
