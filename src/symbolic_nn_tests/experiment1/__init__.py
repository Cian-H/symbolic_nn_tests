LEARNING_RATE = 10e-5


def test(
    train_loss, val_loss, test_loss, accuracy, version, tensorboard=True, wandb=True
):
    from .model import main as test_model

    logger = []
    callbacks = []

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
        from symbolic_nn_tests.callbacks.wandb import ConfusionMatrixCallback

        wandb_logger = WandbLogger(
            project="Symbolic_NN_Tests",
            name=version,
            dir="wandb",
        )
        logger.append(wandb_logger)
        callbacks.append(ConfusionMatrixCallback(class_names=list(map(int, range(10)))))

    test_model(
        logger=logger,
        trainer_callbacks=callbacks,
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=test_loss,
        accuracy=accuracy,
        lr=LEARNING_RATE,
    )

    if wandb:
        _wandb.finish()


def run(tensorboard: bool = True, wandb: bool = True):
    from torchmetrics import Accuracy

    from . import semantic_loss
    from .model import oh_vs_cat_cross_entropy

    test(
        train_loss=oh_vs_cat_cross_entropy,
        val_loss=oh_vs_cat_cross_entropy,
        test_loss=oh_vs_cat_cross_entropy,
        accuracy=Accuracy(task="multiclass", num_classes=10),
        version="cross_entropy",
        tensorboard=tensorboard,
        wandb=wandb,
    )
    test(
        train_loss=semantic_loss.similarity_cross_entropy,
        val_loss=oh_vs_cat_cross_entropy,
        test_loss=oh_vs_cat_cross_entropy,
        accuracy=Accuracy(task="multiclass", num_classes=10),
        version="similarity_cross_entropy",
        tensorboard=tensorboard,
        wandb=wandb,
    )
    test(
        train_loss=semantic_loss.hasline_cross_entropy,
        val_loss=oh_vs_cat_cross_entropy,
        test_loss=oh_vs_cat_cross_entropy,
        accuracy=Accuracy(task="multiclass", num_classes=10),
        version="hasline_cross_entropy",
        tensorboard=tensorboard,
        wandb=wandb,
    )
    test(
        train_loss=semantic_loss.hasloop_cross_entropy,
        val_loss=oh_vs_cat_cross_entropy,
        test_loss=oh_vs_cat_cross_entropy,
        accuracy=Accuracy(task="multiclass", num_classes=10),
        version="hasloop_cross_entropy",
        tensorboard=tensorboard,
        wandb=wandb,
    )
    test(
        train_loss=semantic_loss.multisemantic_cross_entropy,
        val_loss=oh_vs_cat_cross_entropy,
        test_loss=oh_vs_cat_cross_entropy,
        accuracy=Accuracy(task="multiclass", num_classes=10),
        version="multisemantic_cross_entropy",
        tensorboard=tensorboard,
        wandb=wandb,
    )
    test(
        train_loss=semantic_loss.garbage_cross_entropy,
        val_loss=oh_vs_cat_cross_entropy,
        test_loss=oh_vs_cat_cross_entropy,
        accuracy=Accuracy(task="multiclass", num_classes=10),
        version="garbage_cross_entropy",
        tensorboard=tensorboard,
        wandb=wandb,
    )
