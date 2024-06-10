from symbolic_nn_tests.experiment2.math import linear_fit, linear_residuals, sech
from torch import nn
import torch


# TODO: implement semantic loss functions
# These functions would enforce relationships we expect to be present in the results
# if the model is performing correctly. The first ones to implement would be:
#     - En should be proportional to molecular mass
#          - Bigger molecule = more dof and more strain
#     - En should be inversely proportional to the differential in electronegativity
#          - Higher electroneg diff = more stable molecule
#          - calc diff as `torch.sum( torch.abs( electronegativity - electronegativity.mean() ) )`
# Best way to enforce this relationship would probably be to apply a multiplier based on
# a normalized sigmoid curve. This would incentivize the model to ensure slope has correct sign
# without creating a reward hack for maximizing/minimizing m and preventing exploding gradients.
# It also allows us to avoid the assumption of linearity: we only care about the direction of
# proportionality.


def positive_slope_linear_loss(out, y):
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
    residual_penalty = (
        (1 / sech(linear_residuals(diff_electronegativity, y_pred, m, c)))
        .abs()
        .float()
        .mean()
    )

    # We also need to calculate a penalty that incentivizes a positive slope. For this, im using softplus
    # to scale the slope as it will penalise negative slopes without just creating a reward hack for
    # maximizing slope.
    slope_penalty = (nn.functional.softplus(-m) + 1).mean()

    # Finally, let's get a smooth L1 loss and scale it based on these penalty functions
    return nn.functional.smooth_l1_loss(y_pred, y) * residual_penalty * slope_penalty

