import torch
import torch.nn.functional as F


class ProjectedSemanticLoss:
    def __init__(self, semantic_matrix):
        self.semantic_matrix = semantic_matrix

    def __call__(self, input_logits, target_cat):
        ce_loss = F.cross_entropy(input_logits, target_cat)

        # Define the gradient projection hook
        def project_gradient(grad_data):
            with torch.enable_grad():
                logits = input_logits.detach().requires_grad_(True)
                penalty_tensor = self.semantic_matrix.to(logits.device)[target_cat]

                # Calculate the raw linear violation
                input_prob = torch.softmax(logits, dim=1)
                raw_violation = (input_prob * penalty_tensor).sum(dim=1)

                # We use Smooth L1 Loss against a target of 0.
                # This acts like L2 (squared) near zero for a smooth gradient fade-out,
                # and L1 (linear) further away to prevent exploding gradients.
                zero_target = torch.zeros_like(raw_violation)
                smoothed_residual = F.smooth_l1_loss(raw_violation, zero_target, reduction='none')

                # The gradient now smoothly vanishes as the model obeys the rules
                sym_grad = torch.autograd.grad(smoothed_residual.sum(), logits)[0]

            # Gradient Surgery (PCGrad)
            dot_product = (grad_data * sym_grad).sum(dim=1, keepdim=True)

            # Only project if conflicting
            is_conflicting = (dot_product < 0).float()

            sym_norm_sq = (sym_grad ** 2).sum(dim=1, keepdim=True) + 1e-8
            projection = is_conflicting * (dot_product / sym_norm_sq) * sym_grad

            surgically_altered_grad_data = grad_data - projection

            # Re-add the symbolic gradient so it learns the manifold
            return surgically_altered_grad_data + sym_grad

        if input_logits.requires_grad:
            input_logits.register_hook(project_gradient)

        return ce_loss

def create_semantic_loss(semantic_matrix):
    return ProjectedSemanticLoss(semantic_matrix)


# NOTE: This similarity matrix defines loss scaling factors for misclassification
#   of numbers from our QMNIST dataset. Visually similar numbers (e.g: 3/8) are
#   penalised less harshly than visually distinct numbers as this mistake is "less
#   mistaken" given our understanding of the visual characteristics of numerals.
#   By using this scaling matric we can inject human knowledge into the model via
#   the loss function, making this an example of a "semantic loss function"
SIMILARITY_MATRIX = torch.tensor(
    [
        [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0],
        [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.0],
        [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.5, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 1.0, 1.0, 1.0],
        [1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
    ]
)
SIMILARITY_MATRIX /= SIMILARITY_MATRIX.sum()  # Normalized to sum of 1

similarity_semantic_loss = create_semantic_loss(SIMILARITY_MATRIX)


# NOTE: The following matrix encodes a simpler semantic penalty for correctly/incorrectly
#   identifying shapes with straight lines in their representation. This can be a bit fuzzy
#   in cases like "9" though.
HASLINE_MATRIX = torch.tensor(
    #    0,    1,     2,     3,    4,    5,     6,    7,    8,     9
    [False, True, False, False, True, True, False, True, False, True]
)
HASLINE_MATRIX = torch.stack([i ^ HASLINE_MATRIX for i in HASLINE_MATRIX]).type(torch.float64)
HASLINE_MATRIX += 1
HASLINE_MATRIX /= HASLINE_MATRIX.sum()  # Normalize to sum of 1

hasline_semantic_loss = create_semantic_loss(HASLINE_MATRIX)


# NOTE: Similarly, we can do the same for closed circular loops in a numeric character
HASLOOP_MATRIX = torch.tensor(
    #   0,     1,     2,     3,     4,     5,    6,     7,    8,    9
    [True, False, False, False, False, False, True, False, True, True]
)
HASLOOP_MATRIX = torch.stack([i ^ HASLOOP_MATRIX for i in HASLOOP_MATRIX]).type(torch.float64)
HASLOOP_MATRIX += 1
HASLOOP_MATRIX /= HASLOOP_MATRIX.sum()  # Normalize to sum of 1

hasloop_semantic_loss = create_semantic_loss(HASLOOP_MATRIX)


# NOTE: We can also combine all of these semantic matrices
MULTISEMANTIC_MATRIX = SIMILARITY_MATRIX * HASLINE_MATRIX * HASLOOP_MATRIX
MULTISEMANTIC_MATRIX /= MULTISEMANTIC_MATRIX.sum()

multisemantic_semantic_loss = create_semantic_loss(MULTISEMANTIC_MATRIX)

# NOTE: As a final test, lets make something similar to tehse but where there's no knowledge,
#   just random data. This will create a benchmark for the effects of this process wothout the
#   "knowledge" component
GARBAGE_MATRIX = torch.rand(10, 10)
GARBAGE_MATRIX /= GARBAGE_MATRIX.sum()

garbage_semantic_loss = create_semantic_loss(GARBAGE_MATRIX)
