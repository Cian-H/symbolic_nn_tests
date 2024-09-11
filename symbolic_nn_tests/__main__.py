import typer
from typing import Optional, Iterable
from typing_extensions import Annotated
from loguru import logger
from . import local, experiment1, experiment2, experiment3


EXPERIMENTS = (local, experiment1, experiment2, experiment3)


def parse_int_or_intiterable(i: Optional[str] = None) -> Iterable[int]:
    if i is None:
        return range(1, len(EXPERIMENTS))
    else:
        return list(map(int, i.replace("local", "0").split(",")))


def main(
    experiments: Annotated[
        Optional[str],
        typer.Option(
            help="A comma separated list of experiments to be run. Defaults to all.",
        ),
    ] = None,
    tensorboard: Annotated[
        bool, typer.Option(help="Whether or not to log via tensorboard")
    ] = True,
    wandb: Annotated[
        bool, typer.Option(help="Whether or not to log via Weights & Biases")
    ] = True,
):
    experiments = parse_int_or_intiterable(experiments)

    import torch

    # Enable tensor cores for compatible GPUs
    for i in range(torch.cuda.device_count()):
        if torch.cuda.get_device_properties(i).major > 6:
            torch.set_float32_matmul_precision("medium")

    for i, n in enumerate(experiments, start=1):
        experiment = EXPERIMENTS[n].run
        logger.info(f"Running Experiment {n} ({i}/{len(experiments)})...")
        experiment(tensorboard=tensorboard, wandb=wandb)


if __name__ == "__main__":
    typer.run(main)
