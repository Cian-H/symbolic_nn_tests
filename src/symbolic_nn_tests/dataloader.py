from pathlib import Path

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import Caltech256
from torchvision.transforms import ToTensor

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets/"


def create_dataset(
    split: tuple[float, float, float] = (0.7, 0.1, 0.2),
    dataset=Caltech256,
    **kwargs,
):
    ds = dataset(str(DATASET_DIR), download=True, transform=ToTensor())

    shuffle = kwargs.pop("shuffle", False)
    shuffle_train = kwargs.pop("shuffle_train", False)

    kwargs["pin_memory"] = True
    kwargs["num_workers"] = 16
    kwargs["prefetch_factor"] = 8
    kwargs["persistent_workers"] = True

    to_shuffle = (shuffle or shuffle_train, shuffle, shuffle)
    train, val, test = (
        DataLoader(i, shuffle=s, **kwargs)
        for i, s in zip(random_split(ds, split), to_shuffle, strict=True)
    )
    return train, val, test


def execute_task(task):
    import torch

    torch.set_num_threads(1)
    return task()
