from pathlib import Path

import scallopy
import torch.nn as nn


class BooleanSemanticLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.scallop = scallopy.Module(
            file=str(Path(__file__).parent / "boolean_constraints.scl"),
            input_mappings={"x_in": range(10)},
            output_mappings={
                "has_loop": [(True,)],
                "has_line": [(True,)],
            },
            provenance="diffaddmultprob",
            retain_graph=True,
        )
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, y_pred, y_true):
        ce_loss = self.cross_entropy(y_pred, y_true)
        y_true_onehot = nn.functional.one_hot(y_true, num_classes=10).float()
        y_pred_probs = self.softmax(y_pred)

        pred_semantics = self.scallop(x_in=y_pred_probs)
        pred_loop = pred_semantics["has_loop"]
        pred_line = pred_semantics["has_line"]

        true_semantics = self.scallop(x_in=y_true_onehot)
        true_loop = true_semantics["has_loop"].detach()
        true_line = true_semantics["has_line"].detach()

        loop_loss = self.mse_loss(pred_loop, true_loop)
        line_loss = self.mse_loss(pred_line, true_line)

        return ce_loss + loop_loss + line_loss


class KleeneSemanticLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.scallop = scallopy.Module(
            file=str(Path(__file__).parent / "kleene_constraints.scl"),
            input_mappings={"x_in": range(10)},
            output_mappings={
                "has_loop_out": [(1,), (-1,), (0,)],
                "has_line_out": [(1,), (-1,), (0,)],
            },
            provenance="diffaddmultprob",
            retain_graph=True,
        )
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, y_pred, y_true):
        ce_loss = self.cross_entropy(y_pred, y_true)
        y_true_onehot = nn.functional.one_hot(y_true, num_classes=10).float()
        y_pred_probs = self.softmax(y_pred)

        # Scallop outputs tensors of shape (batch_size, 3) where:
        # idx 0: P(1) [True]
        # idx 1: P(-1) [False]
        # idx 2: P(0) [Undecidable]
        pred_semantics = self.scallop(x_in=y_pred_probs)
        pred_loop_probs = pred_semantics["has_loop_out"]
        pred_line_probs = pred_semantics["has_line_out"]

        true_semantics = self.scallop(x_in=y_true_onehot)
        true_loop_probs = true_semantics["has_loop_out"].detach()
        true_line_probs = true_semantics["has_line_out"].detach()

        # We run MSE loss directly on the 3D probability vectors.
        # This penalizes the model for blurring epistemic uncertainty (0.5 True, 0.5 False)
        # with ontological ambiguity (1.0 Unknown).
        loop_loss = self.mse_loss(pred_loop_probs, true_loop_probs)
        line_loss = self.mse_loss(pred_line_probs, true_line_probs)

        return ce_loss + loop_loss + line_loss
