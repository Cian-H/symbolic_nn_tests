import torch
from torch import nn


class ExactSemanticLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.softmax = nn.Softmax(dim=1)

        # loop_probs mapping exactly Scallop's weights in Experiment 3.
        # Format: (10 classes, 2 boolean states: True, False)
        self.register_buffer(
            "loop_probs",
            torch.tensor(
                [
                    [0.10, 0.00],  # 0
                    [0.00, 0.10],  # 1
                    [0.00, 0.10],  # 2
                    [0.00, 0.10],  # 3
                    [0.05, 0.05],  # 4
                    [0.00, 0.10],  # 5
                    [0.10, 0.00],  # 6
                    [0.00, 0.10],  # 7
                    [0.10, 0.00],  # 8
                    [0.10, 0.00],  # 9
                ]
            ),
        )

        # line_probs mapping exactly Scallop's weights.
        self.register_buffer(
            "line_probs",
            torch.tensor(
                [
                    [0.00, 0.10],  # 0
                    [0.10, 0.00],  # 1
                    [0.00, 0.10],  # 2
                    [0.05, 0.05],  # 3
                    [0.10, 0.00],  # 4
                    [0.10, 0.00],  # 5
                    [0.00, 0.10],  # 6
                    [0.10, 0.00],  # 7
                    [0.00, 0.10],  # 8
                    [0.05, 0.05],  # 9
                ]
            ),
        )

    def forward(self, y_pred, y_true):
        ce_loss = self.cross_entropy(y_pred, y_true)

        y_pred_probs = self.softmax(y_pred)
        y_true_onehot = nn.functional.one_hot(y_true, num_classes=10).float()

        # Compute exact probability mapping for "has_loop" and "has_line"
        pred_loop = torch.matmul(y_pred_probs, self.loop_probs)
        pred_line = torch.matmul(y_pred_probs, self.line_probs)

        true_loop = torch.matmul(y_true_onehot, self.loop_probs)
        true_line = torch.matmul(y_true_onehot, self.line_probs)

        # Use MSE exactly like Experiment 3's Scallop formulation for 1-to-1 comparison
        loop_loss = self.mse_loss(pred_loop, true_loop)
        line_loss = self.mse_loss(pred_line, true_line)

        return ce_loss + loop_loss + line_loss
