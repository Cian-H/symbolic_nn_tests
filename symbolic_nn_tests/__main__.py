LEARNING_RATE = 10e-5


def qmnist_test(loss_func, version, tensorboard=True, wandb=True):
    from .experiment_1.model import main as test_model

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


def qmnist_experiment():
    from .experiment_1 import semantic_loss
    from torch import nn

    qmnist_test(nn.functional.cross_entropy, "cross_entropy")
    qmnist_test(semantic_loss.similarity_cross_entropy, "similarity_cross_entropy")
    qmnist_test(semantic_loss.hasline_cross_entropy, "hasline_cross_entropy")
    qmnist_test(semantic_loss.hasloop_cross_entropy, "hasloop_cross_entropy")
    qmnist_test(
        semantic_loss.multisemantic_cross_entropy, "multisemantic_cross_entropy"
    )
    qmnist_test(semantic_loss.garbage_cross_entropy, "garbage_cross_entropy")


def main():
    qmnist_experiment()


if __name__ == "__main__":
    main()
