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

    test_model(logger=logger, loss_func=loss_func, lr=LEARNING_RATE)

    if wandb:
        _wandb.finish()


def run(tensorboard: bool = True, wandb: bool = True):
    from . import semantic_loss
    from torch import nn

    test(
        nn.functional.cross_entropy,
        "cross_entropy",
        tensorboard=tensorboard,
        wandb=wandb,
    )
    test(
        semantic_loss.similarity_cross_entropy,
        "similarity_cross_entropy",
        tensorboard=tensorboard,
        wandb=wandb,
    )
    test(
        semantic_loss.hasline_cross_entropy,
        "hasline_cross_entropy",
        tensorboard=tensorboard,
        wandb=wandb,
    )
    test(
        semantic_loss.hasloop_cross_entropy,
        "hasloop_cross_entropy",
        tensorboard=tensorboard,
        wandb=wandb,
    )
    test(
        semantic_loss.multisemantic_cross_entropy,
        "multisemantic_cross_entropy",
        tensorboard=tensorboard,
        wandb=wandb,
    )
    test(
        semantic_loss.garbage_cross_entropy,
        "garbage_cross_entropy",
        tensorboard=tensorboard,
        wandb=wandb,
    )
