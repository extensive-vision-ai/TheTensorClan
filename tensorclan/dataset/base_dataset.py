from abc import ABCMeta, abstractmethod


class BaseDataset(metaclass=ABCMeta):
    @abstractmethod
    def split_dataset(self):
        """
        Returns:
            (Subset, Subset): we need the subsets so we can store the dataset state for reproducibility
        """
        pass

    @staticmethod
    @abstractmethod
    def plot_sample(sample):
        pass

    @staticmethod
    @abstractmethod
    def apply_on_batch(batch, apply_func):
        pass
