import torch
from torch import nn


class LTNSemanticLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        # Real logic connectives (Lukasiewicz t-norm) natively in PyTorch:
        # Equivalence: A <-> B := 1 - |A - B|
        # In LTN, truth values are bounded in [0, 1].

        self.register_buffer(
            "loop_probs",
            torch.tensor(
                [
                    [0.10, 0.00],
                    [0.00, 0.10],
                    [0.00, 0.10],
                    [0.00, 0.10],
                    [0.05, 0.05],
                    [0.00, 0.10],
                    [0.10, 0.00],
                    [0.00, 0.10],
                    [0.10, 0.00],
                    [0.10, 0.00],
                ]
            ),
        )

        self.register_buffer(
            "line_probs",
            torch.tensor(
                [
                    [0.00, 0.10],
                    [0.10, 0.00],
                    [0.00, 0.10],
                    [0.05, 0.05],
                    [0.10, 0.00],
                    [0.10, 0.00],
                    [0.00, 0.10],
                    [0.10, 0.00],
                    [0.00, 0.10],
                    [0.05, 0.05],
                ]
            ),
        )

    def fuzzy_equiv(self, a, b):
        # Lukasiewicz equivalence
        return 1.0 - torch.abs(a - b)

    def forward(self, y_pred, y_true):
        ce_loss = self.cross_entropy(y_pred, y_true)

        y_pred_probs = self.softmax(y_pred)

        pred_loop = torch.matmul(y_pred_probs, self.loop_probs)
        pred_line = torch.matmul(y_pred_probs, self.line_probs)

        true_loop = self.loop_probs[y_true]
        true_line = self.line_probs[y_true]

        # LTN Satisfiability formulation natively in PyTorch:
        # We compute the fuzzy equivalence truth value between prediction and target.
        loop_sat = self.fuzzy_equiv(pred_loop, true_loop)
        line_sat = self.fuzzy_equiv(pred_line, true_line)

        # Loss is 1.0 - satisfiability (we want to maximize truth value)
        loop_loss = 1.0 - loop_sat.mean()
        line_loss = 1.0 - line_sat.mean()

        return ce_loss + loop_loss + line_loss
