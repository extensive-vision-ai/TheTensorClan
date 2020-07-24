from tensorclan.dataset import BaseDataset
from tensorclan.dataset.augmentation import BaseAugmentation

from torchvision import datasets
from torch.utils.data import Subset


class MNIST(BaseDataset):
    test_set: datasets.MNIST
    train_set: datasets.MNIST

    def __init__(self, root: str, transforms: BaseAugmentation):
        self.data_dir = root

        self.train_set = datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=transforms.build_transforms(train=True),
        )

        self.test_set = datasets.MNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=transforms.build_transforms(train=False)
        )

    def split_dataset(self):
        return Subset(self.train_set, indices=range(0, len(self.train_set))), Subset(self.test_set, indices=range(0, len(self.test_set)))

    @staticmethod
    def plot_sample(sample):
        pass

    @staticmethod
    def apply_on_batch(batch, apply_func):
        pass
