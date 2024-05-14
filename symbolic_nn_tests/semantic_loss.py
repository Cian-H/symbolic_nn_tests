import torch


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
).to("cuda")
SIMILARITY_MATRIX /= SIMILARITY_MATRIX.sum()  # Normalized to sum of 1


def similarity_weighted_cross_entropy(input, target):
    ce_loss = torch.nn.functional.cross_entropy(input, target)

    penalty_tensor = SIMILARITY_MATRIX[target.argmax(dim=1)]
    similarity = (target - input).abs()
    similarity_penalty = (similarity * penalty_tensor).sum()
    return ce_loss * similarity_penalty
