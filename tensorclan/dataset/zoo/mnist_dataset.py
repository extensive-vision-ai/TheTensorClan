from tensorclan.dataset import BaseDataset
from tensorclan.dataset.augmentation import BaseAugmentation

from torchvision import datasets
from torch.utils.data import Subset


class MNIST(BaseDataset):
    test_set: datasets.MNIST
    train_set: datasets.MNIST

    def __init__(self, root: str, transforms=None):
        self.data_dir = root
        self.transforms = transforms

    @staticmethod
    def split_dataset(dataset, transforms):

        train_set = datasets.MNIST(
            dataset.data_dir,
            train=True,
            download=True,
            transform=transforms.build_transforms(train=True),
        )

        test_set = datasets.MNIST(
            dataset.data_dir,
            train=False,
            download=True,
            transform=transforms.build_transforms(train=False)
        )
        return Subset(train_set, indices=range(0, len(train_set))), Subset(test_set, indices=range(0, len(test_set)))

    @staticmethod
    def plot_sample(sample):
        pass

