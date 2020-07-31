from typing import Tuple

from tensorclan.dataset.augmentation import  BaseAugmentation

import albumentations as A
import albumentations.pytorch.transforms as AT


class MNISTTransforms(BaseAugmentation):

    mean: Tuple[float] = (0.1307,)
    std: Tuple[float] = (0.3081,)

    def build_train(self):
        train_transforms = A.Compose([
            A.Normalize(mean=self.mean, std=self.std),
            AT.ToTensor()
        ])
        return train_transforms

    def build_test(self):
        test_transforms = A.Compose([
            A.Normalize(mean=self.mean, std=self.std),
            AT.ToTensor()
        ])

        return test_transforms