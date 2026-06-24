import torch
from torch import nn


class LTNKleeneLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        # Kleene Logic mapped to Godel fuzzy logic:
        # True = 1.0
        # False = 0.0
        # Undecidable = 0.5

        # has_loop truth values
        # 0: True, 1: False, 2: False, 3: False, 4: Undecidable,
        # 5: False, 6: True, 7: False, 8: True, 9: True
        self.register_buffer(
            "loop_targets", torch.tensor([1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 1.0])
        )

        # has_line truth values
        # 0: False, 1: True, 2: False, 3: Undecidable, 4: True,
        # 5: True, 6: False, 7: True, 8: False, 9: Undecidable
        self.register_buffer(
            "line_targets", torch.tensor([0.0, 1.0, 0.0, 0.5, 1.0, 1.0, 0.0, 1.0, 0.0, 0.5])
        )

    def godel_equiv(self, a, b):
        # A differentiable approximation for Equivalence in Godel/Zadeh semantics.
        # Strict Godel equivalence is 1 if A==B else min(A,B), which is non-differentiable.
        # In neurosymbolic practice, 1 - |A - B| is the standard relaxation.
        return 1.0 - torch.abs(a - b)

    def forward(self, y_pred, y_true):
        ce_loss = self.cross_entropy(y_pred, y_true)

        y_pred_probs = self.softmax(y_pred)

        # Compute predicted truth value under Zadeh/Godel mappings
        pred_loop = torch.matmul(y_pred_probs, self.loop_targets)
        pred_line = torch.matmul(y_pred_probs, self.line_targets)

        # True expected truth values
        true_loop = self.loop_targets[y_true]
        true_line = self.line_targets[y_true]

        loop_sat = self.godel_equiv(pred_loop, true_loop)
        line_sat = self.godel_equiv(pred_line, true_line)

        # Maximize satisfiability -> minimize 1 - sat
        loop_loss = 1.0 - loop_sat.mean()
        line_loss = 1.0 - line_sat.mean()

        return ce_loss + loop_loss + line_loss
