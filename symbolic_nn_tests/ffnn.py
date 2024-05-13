from torch import nn


model = nn.Sequential(
    nn.Flatten(1, -1),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
    nn.Softmax(dim=1),
)


def main():
    from torchvision.datasets import QMNIST
    import lightning as L

    from .dataloader import get_dataset
    from .trainer import Trainer

    train, test, val = get_dataset(dataset=QMNIST)
    training_model = Trainer(model)
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model=training_model, train_dataloaders=train, val_dataloaders=val)
