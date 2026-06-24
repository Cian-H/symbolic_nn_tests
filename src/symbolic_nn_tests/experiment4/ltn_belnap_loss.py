import torch
from torch import nn


class LTNBelnapLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        # Fuzzy Belnap Logic Mapping (t, f):
        # True = (1.0, 0.0)
        # False = (0.0, 1.0)
        # Undecidable (Both/Conflict) = (1.0, 1.0)
        # Uncertain (None/Unknown) = (0.0, 0.0)

        # has_loop truth states
        # 0: True, 1: False, 2: False, 3: False, 4: Undecidable,
        # 5: False, 6: True, 7: False, 8: True, 9: True
        self.register_buffer(
            "loop_t_targets",
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
        )
        self.register_buffer(
            "loop_f_targets",
            torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]),
        )

        # has_line truth states
        # 0: False, 1: True, 2: False, 3: Undecidable, 4: True,
        # 5: True, 6: False, 7: True, 8: False, 9: Undecidable
        self.register_buffer(
            "line_t_targets",
            torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        )
        self.register_buffer(
            "line_f_targets",
            torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
        )

    def fuzzy_equiv(self, pred_t, pred_f, target_t, target_f):
        # Lukasiewicz bi-implication applied over bilattice independently
        sat_t = 1.0 - torch.abs(pred_t - target_t)
        sat_f = 1.0 - torch.abs(pred_f - target_f)
        # Combine evidence satisfaction (avg/min, avg smoother for gradients)
        return 0.5 * (sat_t + sat_f)

    def forward(self, y_pred, y_true):
        ce_loss = self.cross_entropy(y_pred, y_true)

        y_pred_probs = self.softmax(y_pred)

        # Compute predicted truth and falsity evidence independently
        pred_loop_t = torch.matmul(y_pred_probs, self.loop_t_targets)
        pred_loop_f = torch.matmul(y_pred_probs, self.loop_f_targets)

        pred_line_t = torch.matmul(y_pred_probs, self.line_t_targets)
        pred_line_f = torch.matmul(y_pred_probs, self.line_f_targets)

        # Get target evidence
        true_loop_t = self.loop_t_targets[y_true]
        true_loop_f = self.loop_f_targets[y_true]

        true_line_t = self.line_t_targets[y_true]
        true_line_f = self.line_f_targets[y_true]

        loop_sat = self.fuzzy_equiv(
            pred_loop_t,
            pred_loop_f,
            true_loop_t,
            true_loop_f,
        )
        line_sat = self.fuzzy_equiv(
            pred_line_t,
            pred_line_f,
            true_line_t,
            true_line_f,
        )

        # Loss is 1 - satisfiability
        loop_loss = 1.0 - loop_sat.mean()
        line_loss = 1.0 - line_sat.mean()

        return ce_loss + loop_loss + line_loss
