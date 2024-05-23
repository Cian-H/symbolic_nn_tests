import typer
from typing import Optional, Iterable
from typing_extensions import Annotated
from . import experiment1


EXPERIMENTS = (experiment1,)


def parse_int_or_intiterable(i: Optional[str]) -> Iterable[int]:
    return range(1, len(EXPERIMENTS) + 1) if i is None else map(int, i.split(","))


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
    experiment_indeces = (i - 1 for i in experiments)
    experiment_funcs = [EXPERIMENTS[i].run for i in experiment_indeces]

    for experiment in experiment_funcs:
        experiment(tensorboard=tensorboard, wandb=wandb)


if __name__ == "__main__":
    typer.run(main)
