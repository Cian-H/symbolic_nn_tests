LEARNING_RATE = 10e-5


def test(loss_func, version, tensorboard=True, wandb=True):
    from .model import main as test_model

    logger = []

    if tensorboard:
        from lightning.pytorch.loggers import TensorBoardLogger

        tb_logger = TensorBoardLogger(
            save_dir=".",
            name="logs/comparison",
            version=version,
        )
        logger.append(tb_logger)

    if wandb:
        import wandb as _wandb
        from lightning.pytorch.loggers import WandbLogger

        wandb_logger = WandbLogger(
            project="Symbolic_NN_Tests",
            name=version,
            dir="wandb",
        )
        logger.append(wandb_logger)

    test_model(logger=logger, loss_func=loss_func)

    if wandb:
        _wandb.finish()


def run(tensorboard: bool = True, wandb: bool = True):
    from .model import unpacking_mse_loss
    from . import semantic_loss

    test(
        unpacking_mse_loss,
        "mse_loss",
        tensorboard=tensorboard,
        wandb=wandb,
    )
    test(
        semantic_loss.positive_slope_linear_loss,
        "positive_slope_linear_loss",
        tensorboard=tensorboard,
        wandb=wandb,
    )
