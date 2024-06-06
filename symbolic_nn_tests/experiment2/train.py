from symbolic_nn_tests.train import TrainingWrapper


class SemanticModuleTrainingWrapper(TrainingWrapper):
    def __init__(self, model, *args, loss_func0, loss_func1, loss_agg, **kwargs):
        assert len(args) == 0

        super().__init__(model, **kwargs)
        self.loss_func0 = loss_func0
        self.loss_func1 = loss_func1
        self.loss_agg = loss_agg

    def _forward_step(self, batch, batch_idx, label=""):
        x, y = batch
        y_pred, y0, y1 = self.model(x)
        loss = self.loss_func(y_pred, y)
        loss0 = self.loss_func0(y0, x)
        loss1 = self.loss_func1(y1, x)
        self.log(f"{label}{'_' if label else ''}loss", loss)
        return self.loss_agg(loss, loss0, loss1)
