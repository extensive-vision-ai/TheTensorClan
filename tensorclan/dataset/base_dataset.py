from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    @staticmethod
    @abstractmethod
    def split_dataset(dataset, transforms):
        """
        Returns:
            (Subset, Subset): we need the subsets so we can store the dataset state for reproducibility
        """
        pass

    @staticmethod
    @abstractmethod
    def plot_sample(sample):
        pass
