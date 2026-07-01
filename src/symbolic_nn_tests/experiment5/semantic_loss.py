import torch
import torch.nn.functional as F


class SemanticCrossEntropy:
    def __init__(self, semantic_matrix, alpha=10.0, temperature=1.0):
        self.semantic_matrix = semantic_matrix
        self.alpha = alpha
        self.temperature = temperature

    def __call__(self, input_logits, target_cat):
        target = F.one_hot(target_cat, num_classes=10).float()

        # Get Cross Entropy PER SAMPLE (Shape: [Batch_Size])
        # reduction='none' stops PyTorch from averaging it immediately
        ce_loss_per_sample = F.cross_entropy(input_logits, target, reduction="none")

        # Get Semantic Penalty PER SAMPLE (Shape: [Batch_Size])
        penalty_tensor = self.semantic_matrix.to(target.device)[target_cat]
        input_prob = torch.softmax(input_logits, dim=1)
        abs_diff = (target - input_prob).abs()

        semantic_penalty_per_sample = (abs_diff * penalty_tensor).mean(dim=1)
        weighted_penalty_per_sample = self.alpha * semantic_penalty_per_sample

        scaled_ce = ce_loss_per_sample / self.temperature
        scaled_penalty = weighted_penalty_per_sample / self.temperature

        combined_loss_per_sample = self.temperature * torch.logaddexp(scaled_ce, scaled_penalty)
        return combined_loss_per_sample.mean()

def create_semantic_cross_entropy(semantic_matrix):
    return SemanticCrossEntropy(semantic_matrix)


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

similarity_cross_entropy = create_semantic_cross_entropy(SIMILARITY_MATRIX)


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

hasline_cross_entropy = create_semantic_cross_entropy(HASLINE_MATRIX)


# NOTE: Similarly, we can do the same for closed circular loops in a numeric character
HASLOOP_MATRIX = torch.tensor(
    #   0,     1,     2,     3,     4,     5,    6,     7,    8,    9
    [True, False, False, False, False, False, True, False, True, True]
)
HASLOOP_MATRIX = torch.stack([i ^ HASLOOP_MATRIX for i in HASLOOP_MATRIX]).type(torch.float64)
HASLOOP_MATRIX += 1
HASLOOP_MATRIX /= HASLOOP_MATRIX.sum()  # Normalize to sum of 1

hasloop_cross_entropy = create_semantic_cross_entropy(HASLOOP_MATRIX)


# NOTE: We can also combine all of these semantic matrices
MULTISEMANTIC_MATRIX = SIMILARITY_MATRIX * HASLINE_MATRIX * HASLOOP_MATRIX
MULTISEMANTIC_MATRIX /= MULTISEMANTIC_MATRIX.sum()

multisemantic_cross_entropy = create_semantic_cross_entropy(MULTISEMANTIC_MATRIX)

# NOTE: As a final test, lets make something similar to tehse but where there's no knowledge,
#   just random data. This will create a benchmark for the effects of this process wothout the
#   "knowledge" component
GARBAGE_MATRIX = torch.rand(10, 10)
GARBAGE_MATRIX /= GARBAGE_MATRIX.sum()

garbage_cross_entropy = create_semantic_cross_entropy(GARBAGE_MATRIX)
