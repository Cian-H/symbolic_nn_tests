LEARNING_RATE = 10e-5


def test(train_loss, val_loss, test_loss, version, tensorboard=True, wandb=True):
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

        if isinstance(wandb, WandbLogger):
            wandb_logger = wandb
        else:
            wandb_logger = WandbLogger(
                project="Symbolic_NN_Tests",
                name=version,
                dir="wandb",
                log_model="all",
            )
        logger.append(wandb_logger)

    test_model(
        logger=logger,
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=test_loss,
        lr=LEARNING_RATE,
    )

    if wandb:
        _wandb.finish()


def run(tensorboard: bool = True, wandb: bool = True):
    from .model import unpacking_smooth_l1_loss
    from . import semantic_loss

    test(
        train_loss=unpacking_smooth_l1_loss,
        val_loss=unpacking_smooth_l1_loss,
        test_loss=unpacking_smooth_l1_loss,
        version="smooth_l1_loss",
        tensorboard=tensorboard,
        wandb=wandb,
    )

    version = "positive_slope_linear_loss"
    if wandb:
        from lightning.pytorch.loggers import WandbLogger

        wandb_logger = WandbLogger(
            project="Symbolic_NN_Tests",
            name=version,
            dir="wandb",
            log_model="all",
        )
    else:
        wandb_logger = wandb

    test(
        train_loss=semantic_loss.positive_slope_linear_loss(wandb_logger, version),
        val_loss=unpacking_smooth_l1_loss,
        test_loss=unpacking_smooth_l1_loss,
        version=version,
        tensorboard=tensorboard,
        wandb=wandb_logger,
    )
