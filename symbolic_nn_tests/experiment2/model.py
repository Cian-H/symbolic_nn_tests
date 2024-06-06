from functools import lru_cache
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, return_module_y=False):
        super().__init__()

        self.return_module_y = return_module_y

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
        y0 = self.encode_x0(x0)
        y1 = self.encode_x1(x1)
        y = torch.cat([y0, y1], dim=1)
        y = self.ff(y)
        if self.return_module_y:
            return x, (y, y0, y1)
        else:
            return x, y


# This is just a quick, lazy way to ensure all models are trained on the same dataset
@lru_cache(maxsize=1)
def get_singleton_dataset():
    from symbolic_nn_tests.dataloader import create_dataset
    from symbolic_nn_tests.experiment2.dataset import collate, pubchem

    return create_dataset(
        dataset=pubchem, collate_fn=collate, batch_size=128, shuffle=True
    )


def smooth_l1_loss(out, y):
    _, y_pred = out
    return nn.functional.smooth_l1_loss(y_pred, y)


def sech(x):
    return torch.reciprocal(torch.cosh(x))


def linear_fit(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    cov_xy = torch.mean(x * y) - (mean_x * mean_y)
    var_x = torch.mean(x * x) - (mean_x * mean_x)
    m = cov_xy / var_x
    c = mean_y - (m * mean_x)
    return m, c


def line(x, m, c):
    return (m * x) + c


def linear_residuals(x, y, m, c):
    return y - line(x, m, c)


def semantic_loss(x, y_pred, w, a):
    m, c = linear_fit(x, y_pred)
    residuals = linear_residuals(x, y_pred, m, c)
    scaled_residuals = residuals * sech(w * x)
    slope_penalty = torch.nn.functional.softmax(a * m, dim=0)
    loss = torch.mean(scaled_residuals**2) + torch.mean(slope_penalty)
    return loss


def loss(out, y):
    x, y_pred = out
    x0, x1 = x

    # Here, we want to make semantic use of the differential electronegativity of the molecule
    # so start by calculating that
    mean_electronegativities = torch.tensor(
        [i[:, 3].mean() for i in x0], dtype=torch.float32
    ).to(y_pred.device)
    diff_electronegativity = (
        torch.tensor(
            [
                (i[:, 3] - mean).abs().sum()
                for i, mean in zip(x0, mean_electronegativities)
            ],
            dtype=torch.float32,
        )
        * 4.0
    ).to(y_pred.device)

    # Then, we need to get a linear best fit on that. Our semantic info is based on a graph of
    # En (y) vs differential electronegativity on the x vs y axes, so y_pred is y here
    m, c = linear_fit(diff_electronegativity, y_pred)

    # To start with, we want to calculate a penalty based on deviation from a linear relationship
    # Scaling is being based on 1/sech(w*r) as this increases multiplier as deviation grows.
    # `w` was selected based on noting that the residual spread before eneg scaling was about 25;
    # enegs were normalised as x/4, so we want to incentivize a spread of about 25/4~=6, and w=0.2
    # causes the penalty function to cross 2 at just over 6. Yes, that's a bit arbitrary but we're
    # just steering the model not applying hard constraints to it shold be fine.
    residual_penalty = (
        (
            linear_residuals(diff_electronegativity, y_pred, m, c)
            / sech(0.2 * diff_electronegativity)
        )
        .abs()
        .float()
        .mean()
    )

    # We also need to calculate a penalty that incentivizes a positive slope. For this, im using softmax
    # to scale the slope as it will penalise negative slopes while not just creating a reward hack for
    # maximizing slope. The softmax function approximates 1 from about 5 onwards, so if we multiply m by
    # 500, then our penalty should be almost minimised for any slope above 0.01 and maximised below 0.01.
    # This should suffice for incentivizing the model to favour positive slopes.
    slope_penalty = (torch.nn.functional.softmax(-m * 500.0) + 1).mean()

    # Finally, let's get a smooth L1 loss and scale it based on these penalty functions
    return nn.functional.smooth_l1_loss(y_pred, y) * residual_penalty * slope_penalty


# def main(loss_func=smooth_l1_loss, logger=None, **kwargs):
def main(loss_func=loss, logger=None, **kwargs):
    import lightning as L

    from symbolic_nn_tests.train import TrainingWrapper

    if logger is None:
        from lightning.pytorch.loggers import TensorBoardLogger

        logger = TensorBoardLogger(save_dir=".", name="logs/ffnn")

    train, val, test = get_singleton_dataset()
    lmodel = TrainingWrapper(Model(), loss_func=loss_func)
    lmodel.configure_optimizers(optimizer=torch.optim.NAdam, **kwargs)
    trainer = L.Trainer(max_epochs=10, logger=logger, num_sanity_val_steps=0)
    trainer.fit(model=lmodel, train_dataloaders=train, val_dataloaders=val)
    trainer.test(dataloaders=test)


if __name__ == "__main__":
    main()
