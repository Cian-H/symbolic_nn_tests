from .model import main


if __name__ == "__main__":
    from lightning.pytorch.loggers import TensorBoardLogger

    logger = TensorBoardLogger(save_dir=".", name="logs/ffnn")
    main(logger)
