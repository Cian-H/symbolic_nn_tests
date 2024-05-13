from pathlib import Path
from torchvision.datasets import Caltech256
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torch.utils.data import BatchSampler


PROJECT_ROOT = Path(__file__).parent.parent


def get_dataset(
    split: (float, float, float) = (0.7, 0.1, 0.2),
    batch_size: int = 128,
    drop_last: bool = False,
):
    ds = Caltech256(PROJECT_ROOT / "datasets/", download=True, transform=ToTensor())
    train, test, val = (
        BatchSampler(i, batch_size, drop_last) for i in random_split(ds, split)
    )
    return train, test, val
