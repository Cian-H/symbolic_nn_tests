from typing import Annotated

import typer
from loguru import logger

from . import experiment1, experiment2, experiment3, experiment4, experiment5, local

EXPERIMENTS = (local, experiment1, experiment2, experiment3, experiment4, experiment5)


def parse_int_or_intiterable(i: str | None = None) -> list[int]:
    if i is None:
        return list(range(1, len(EXPERIMENTS)))
    return list(map(int, i.replace("local", "0").split(",")))


def main(
    experiments: Annotated[
        str | None,
        typer.Option(
            help="A comma separated list of experiments to be run. Defaults to all.",
        ),
    ] = None,
    tensorboard: Annotated[bool, typer.Option(help="Whether or not to log via tensorboard")] = True,
    wandb: Annotated[bool, typer.Option(help="Whether or not to log via Weights & Biases")] = True,
):
    experiments_to_run = parse_int_or_intiterable(experiments)

    import torch

    torch.backends.cudnn.benchmark = True

    # Enable tensor cores for compatible GPUs
    for i in range(torch.cuda.device_count()):
        if torch.cuda.get_device_properties(i).major > 6:
            torch.set_float32_matmul_precision("medium")

    import functools
    from collections.abc import Callable
    from typing import Any

    tasks: list[Callable[[], Any]] = []

    # Helper to capture tasks instead of running them sequentially
    def capture_tasks(module):
        original_test = module.test

        def mock_test(*args, **kwargs):
            mock = functools.partial(original_test, *args, **kwargs)
            tasks.append(mock)

        module.test = mock_test
        module.run(tensorboard, wandb)
        module.test = original_test

    for n in experiments_to_run:
        experiment = EXPERIMENTS[n]
        if hasattr(experiment, "test"):
            capture_tasks(experiment)
        else:
            tasks.append(functools.partial(experiment.run, tensorboard=tensorboard, wandb=wandb))

    logger.info(f"Running {len(tasks)} models sequentially...")
    for task in tasks:
        task()

    from symbolic_nn_tests.plot_metrics import generate_and_log_summary_plots

    generate_and_log_summary_plots()


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    typer.run(main)
