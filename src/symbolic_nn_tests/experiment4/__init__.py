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
            name="logs/experiment4",
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
    from .exact_kleene_loss import ExactKleeneSemanticLoss
    from .exact_loss import ExactSemanticLoss
    from .ltn_kleene_loss import LTNKleeneLoss
    from .ltn_loss import LTNSemanticLoss

    loss_fn_exact = ExactSemanticLoss()
    test(
        train_loss=loss_fn_exact,
        val_loss=loss_fn_exact,
        test_loss=loss_fn_exact,
        version="exact_semantic_loss",
        tensorboard=tensorboard,
        wandb=wandb,
    )

    loss_fn_ltn = LTNSemanticLoss()
    test(
        train_loss=loss_fn_ltn,
        val_loss=loss_fn_ltn,
        test_loss=loss_fn_ltn,
        version="ltn_semantic_loss",
        tensorboard=tensorboard,
        wandb=wandb,
    )

    loss_fn_exact_kleene = ExactKleeneSemanticLoss()
    test(
        train_loss=loss_fn_exact_kleene,
        val_loss=loss_fn_exact_kleene,
        test_loss=loss_fn_exact_kleene,
        version="exact_kleene_semantic_loss",
        tensorboard=tensorboard,
        wandb=wandb,
    )

    loss_fn_ltn_kleene = LTNKleeneLoss()
    test(
        train_loss=loss_fn_ltn_kleene,
        val_loss=loss_fn_ltn_kleene,
        test_loss=loss_fn_ltn_kleene,
        version="ltn_kleene_semantic_loss",
        tensorboard=tensorboard,
        wandb=wandb,
    )

    from .ltn_belnap_loss import LTNBelnapLoss

    loss_fn_ltn_belnap = LTNBelnapLoss()
    test(
        train_loss=loss_fn_ltn_belnap,
        val_loss=loss_fn_ltn_belnap,
        test_loss=loss_fn_ltn_belnap,
        version="ltn_belnap_semantic_loss",
        tensorboard=tensorboard,
        wandb=wandb,
    )
