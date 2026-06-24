import torch
from torch import nn


class ExactKleeneSemanticLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.softmax = nn.Softmax(dim=1)

        # loop_probs mapping for [P(True), P(False), P(Undecidable)]
        self.register_buffer(
            "loop_probs",
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],  # 0
                    [0.0, 1.0, 0.0],  # 1
                    [0.0, 1.0, 0.0],  # 2
                    [0.0, 1.0, 0.0],  # 3
                    [0.0, 0.0, 1.0],  # 4
                    [0.0, 1.0, 0.0],  # 5
                    [1.0, 0.0, 0.0],  # 6
                    [0.0, 1.0, 0.0],  # 7
                    [1.0, 0.0, 0.0],  # 8
                    [1.0, 0.0, 0.0],  # 9
                ]
            ),
        )

        self.register_buffer(
            "line_probs",
            torch.tensor(
                [
                    [0.0, 1.0, 0.0],  # 0
                    [1.0, 0.0, 0.0],  # 1
                    [0.0, 1.0, 0.0],  # 2
                    [0.0, 0.0, 1.0],  # 3
                    [1.0, 0.0, 0.0],  # 4
                    [1.0, 0.0, 0.0],  # 5
                    [0.0, 1.0, 0.0],  # 6
                    [1.0, 0.0, 0.0],  # 7
                    [0.0, 1.0, 0.0],  # 8
                    [0.0, 0.0, 1.0],  # 9
                ]
            ),
        )

    def forward(self, y_pred, y_true):
        ce_loss = self.cross_entropy(y_pred, y_true)

        y_pred_probs = self.softmax(y_pred)
        y_true_onehot = nn.functional.one_hot(y_true, num_classes=10).float()

        pred_loop = torch.matmul(y_pred_probs, self.loop_probs)
        pred_line = torch.matmul(y_pred_probs, self.line_probs)

        true_loop = torch.matmul(y_true_onehot, self.loop_probs)
        true_line = torch.matmul(y_true_onehot, self.line_probs)

        loop_loss = self.mse_loss(pred_loop, true_loop)
        line_loss = self.mse_loss(pred_line, true_line)

        return ce_loss + loop_loss + line_loss
