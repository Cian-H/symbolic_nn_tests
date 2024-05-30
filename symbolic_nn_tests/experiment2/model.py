from functools import lru_cache
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.x0_encoder = nn.TransformerEncoderLayer(7, 7)
        self.x1_encoder = nn.TransformerEncoderLayer(10, 10)
        self.encode_x0 = self.create_xval_encoding_fn(self.x0_encoder)
        self.encode_x1 = self.create_xval_encoding_fn(self.x1_encoder)
        self.ff = nn.Sequential(
            nn.Linear(17, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    @staticmethod
    def create_xval_encoding_fn(layer):
        def encoding_fn(xbatch):
            return torch.stack([layer(x)[-1] for x in xbatch])

        return encoding_fn

    def forward(self, x):
        x0, x1 = x
        x0 = self.encode_x0(x0)
        x1 = self.encode_x1(x1)
        x = torch.cat([x0, x1], dim=1)
        y = self.ff(x)
        return y


# This is just a quick, lazy way to ensure all models are trained on the same dataset
@lru_cache(maxsize=1)
def get_singleton_dataset():
    from symbolic_nn_tests.dataloader import create_dataset
    from symbolic_nn_tests.experiment2.dataset import collate, pubchem

    return create_dataset(
        dataset=pubchem, collate_fn=collate, batch_size=128, shuffle=True
    )


def main(loss_func=nn.functional.smooth_l1_loss, logger=None, **kwargs):
    import lightning as L

    from symbolic_nn_tests.train import TrainingWrapper

    if logger is None:
        from lightning.pytorch.loggers import TensorBoardLogger

        logger = TensorBoardLogger(save_dir=".", name="logs/ffnn")

    train, val, test = get_singleton_dataset()
    lmodel = TrainingWrapper(Model(), loss_func=loss_func)
    lmodel.configure_optimizers(optimizer=torch.optim.NAdam, **kwargs)
    trainer = L.Trainer(max_epochs=20, logger=logger)
    trainer.fit(model=lmodel, train_dataloaders=train, val_dataloaders=val)
    trainer.test(dataloaders=test)


if __name__ == "__main__":
    main()
