LEARNING_RATE = 10e-5


def run_test(loss_func, version):
    from .model import main as test_model
    from lightning.pytorch.loggers import TensorBoardLogger

    logger = TensorBoardLogger(
        save_dir=".",
        name="logs/comparison",
        version=version,
    )
    test_model(lr=LEARNING_RATE)
    # test_model(logger=logger, loss_func=loss_func, lr=LEARNING_RATE)


def main():
    from . import semantic_loss
    from torch import nn

    run_test(nn.functional.cross_entropy, "cross_entropy")
    # run_test(semantic_loss.similarity_cross_entropy, "similarity_cross_entropy")
    # run_test(semantic_loss.hasline_cross_entropy, "hasline_cross_entropy")
    # run_test(semantic_loss.hasloop_cross_entropy, "hasloop_cross_entropy")
    # run_test(semantic_loss.multisemantic_cross_entropy, "multisemantic_cross_entropy")


if __name__ == "__main__":
    main()
