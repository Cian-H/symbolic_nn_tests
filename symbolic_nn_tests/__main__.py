from typing import Optional, Iterable
import experiment1


EXPERIMENTS = (experiment1,)


def main(
    experiments: Optional[int | Iterable[int]] = None,
    tensorboard: bool = True,
    wandb: bool = True,
):
    if experiments is None:
        experiments = range(1, len(EXPERIMENTS) + 1)
    elif not isinstance(experiments, Iterable):
        experiments = (experiments,)

    experiment_indeces = (i - 1 for i in experiments)
    experiment_funcs = [EXPERIMENTS[i].run for i in experiment_indeces]

    for experiment in experiment_funcs:
        experiment(tensorboard=tensorboard, wandb=wandb)


if __name__ == "__main__":
    main()
