def main():
    from .model import main as test_model
    from . import semantic_loss
    from lightning.pytorch.loggers import TensorBoardLogger
    from torch import nn

    logger = TensorBoardLogger(
        save_dir=".",
        name="logs/comparison",
        version="cross_entropy",
    )
    test_model(logger=logger, loss_func=nn.functional.cross_entropy)

    logger = TensorBoardLogger(
        save_dir=".",
        name="logs/comparison",
        version="similarity_weighted_cross_entropy",
    )
    test_model(logger=logger, loss_func=semantic_loss.similarity_weighted_cross_entropy)


if __name__ == "__main__":
    main()
