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
    train, val, test = (DataLoader(i, **kwargs) for i in random_split(ds, split))
    return train, val, test
