import typer
from typing import Optional, Iterable
from typing_extensions import Annotated
from loguru import logger
from . import experiment1, experiment2, experiment3


EXPERIMENTS = (experiment1, experiment2, experiment3)


def parse_int_or_intiterable(i: Optional[str]) -> Iterable[int]:
    return range(1, len(EXPERIMENTS) + 1) if i is None else list(map(int, i.split(",")))


def main(
    experiments: Annotated[
        Optional[str],
        typer.Option(
            parser=parse_int_or_intiterable,
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
    import torch

    # Enable tensor cores for compatible GPUs
    for i in range(torch.cuda.device_count()):
        if torch.cuda.get_device_properties(i).major > 6:
            torch.set_float32_matmul_precision("medium")

    for i, n in enumerate(experiments, start=1):
        j = n - 1
        experiment = EXPERIMENTS[j].run
        logger.info(f"Running Experiment {n} ({i}/{len(experiments)})...")
        experiment(tensorboard=tensorboard, wandb=wandb)


if __name__ == "__main__":
    typer.run(main)
