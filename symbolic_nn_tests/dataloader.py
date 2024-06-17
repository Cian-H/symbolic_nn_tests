from pathlib import Path
from torchvision.datasets import Caltech256
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split


PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets/"


def create_dataset(
    split: (float, float, float) = (0.7, 0.1, 0.2),
    dataset=Caltech256,
    **kwargs,
):
    ds = dataset(DATASET_DIR, download=True, transform=ToTensor())
    shuffle = kwargs.pop("shuffle", False)
    shuffle_train = kwargs.pop("shuffle_train", False)
    to_shuffle = (shuffle or shuffle_train, shuffle, shuffle)
    train, val, test = (
        DataLoader(i, shuffle=s, **kwargs)
        for i, s in zip(random_split(ds, split), to_shuffle)
    )
    return train, val, test
