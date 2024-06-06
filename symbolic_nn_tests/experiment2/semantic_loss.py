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
