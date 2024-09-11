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

        wandb_logger = WandbLogger(
            project="Symbolic_NN_Tests",
            name=version,
            dir="wandb",
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
    from . import semantic_loss
    from .model import oh_vs_cat_cross_entropy

    test(
        train_loss=oh_vs_cat_cross_entropy,
        val_loss=oh_vs_cat_cross_entropy,
        test_loss=oh_vs_cat_cross_entropy,
        version="cross_entropy",
        tensorboard=tensorboard,
        wandb=wandb,
    )
    test(
        train_loss=semantic_loss.similarity_cross_entropy,
        val_loss=oh_vs_cat_cross_entropy,
        test_loss=oh_vs_cat_cross_entropy,
        version="similarity_cross_entropy",
        tensorboard=tensorboard,
        wandb=wandb,
    )
    test(
        train_loss=semantic_loss.hasline_cross_entropy,
        val_loss=oh_vs_cat_cross_entropy,
        test_loss=oh_vs_cat_cross_entropy,
        version="hasline_cross_entropy",
        tensorboard=tensorboard,
        wandb=wandb,
    )
    test(
        train_loss=semantic_loss.hasloop_cross_entropy,
        val_loss=oh_vs_cat_cross_entropy,
        test_loss=oh_vs_cat_cross_entropy,
        version="hasloop_cross_entropy",
        tensorboard=tensorboard,
        wandb=wandb,
    )
    test(
        train_loss=semantic_loss.multisemantic_cross_entropy,
        val_loss=oh_vs_cat_cross_entropy,
        test_loss=oh_vs_cat_cross_entropy,
        version="multisemantic_cross_entropy",
        tensorboard=tensorboard,
        wandb=wandb,
    )
    test(
        train_loss=semantic_loss.garbage_cross_entropy,
        val_loss=oh_vs_cat_cross_entropy,
        test_loss=oh_vs_cat_cross_entropy,
        version="garbage_cross_entropy",
        tensorboard=tensorboard,
        wandb=wandb,
    )
