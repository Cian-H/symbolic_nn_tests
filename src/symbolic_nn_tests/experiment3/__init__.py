LEARNING_RATE = 10e-5


def test(
    train_loss,
    val_loss,
    test_loss,
    version,
    tensorboard=True,
    wandb=True,
):
    from .model import main as test_model

    logger = []
    callbacks = []

    if tensorboard:
        from lightning.pytorch.loggers import TensorBoardLogger

        from symbolic_nn_tests.callbacks.tensorboard import TensorBoardConfusionMatrixCallback

        tb_logger = TensorBoardLogger(
            save_dir=".",
            name="logs/experiment3",
            version=version,
        )
        logger.append(tb_logger)
        callbacks.append(TensorBoardConfusionMatrixCallback(class_names=list(map(str, range(10)))))

    if wandb:
        from lightning.pytorch.loggers import WandbLogger

        import wandb as _wandb
        from symbolic_nn_tests.callbacks.wandb import ConfusionMatrixCallback

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
        callbacks.append(ConfusionMatrixCallback(class_names=list(map(int, range(10)))))

    test_model(
        logger=logger,
        trainer_callbacks=callbacks,
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=test_loss,
        lr=LEARNING_RATE,
    )

    if wandb:
        _wandb.finish()


def run(tensorboard: bool = True, wandb: bool = True):
    from .semantic_loss import BooleanSemanticLoss, KleeneSemanticLoss

    # loss_fn_boolean = BooleanSemanticLoss()
    # test(
    #     train_loss=loss_fn_boolean,
    #     val_loss=loss_fn_boolean,
    #     test_loss=loss_fn_boolean,
    #     version="differentiable_boolean_constraints",
    #     tensorboard=tensorboard,
    #     wandb=wandb,
    # )

    loss_fn_kleene = KleeneSemanticLoss()
    test(
        train_loss=loss_fn_kleene,
        val_loss=loss_fn_kleene,
        test_loss=loss_fn_kleene,
        version="differentiable_kleene_constraints",
        tensorboard=tensorboard,
        wandb=wandb,
    )
