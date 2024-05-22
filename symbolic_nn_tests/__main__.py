LEARNING_RATE = 10e-5


def qmnist_test(loss_func, version):
    from .qmnist.model import main as test_model
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.loggers import WandbLogger
    import wandb

    tb_logger = TensorBoardLogger(
        save_dir=".",
        name="logs/comparison",
        version=version,
    )
    # wandb_logger = WandbLogger(
    #     project="Symbolic_NN_Tests",
    #     name=version,
    #     dir="wandb",
    # )
    logger = [
        tb_logger,
    ]  # wandb_logger]
    test_model(logger=logger, loss_func=loss_func, lr=LEARNING_RATE)
    wandb.finish()


def qmnist_experiment():
    from .qmnist import semantic_loss
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
