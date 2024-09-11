from symbolic_nn_tests.train import TrainingWrapper as _TrainingWrapper
import torch
from skopt import Optimizer
from skopt.learning import RandomForestRegressor


class TrainingWrapper(_TrainingWrapper):
    def __init__(self, *args, loss_rate_target=-10, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_optimizer = Optimizer(
            [(0.0, 512.0), (0.0, 1024.0), (0.0, 512.0)],
            base_estimator=RandomForestRegressor(
                n_jobs=-1,
            ),
            n_initial_points=10,
            model_queue_size=10,
            acq_func="gp_hedge",
        )
        self.loss_rate_target = torch.tensor(loss_rate_target).float()
        self.losses = []

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        self.adjust_train_loss(loss)
        return loss

    def adjust_train_loss(self, loss):
        self.loss_optimizer.tell(self.train_loss.params, loss.item())
        self.train_loss.params = self.loss_optimizer.ask()
