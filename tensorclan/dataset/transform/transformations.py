from abc import ABCMeta, abstractmethod

import numpy as np


class BaseTransform(metaclass=ABCMeta):
    def build_transforms(self, train: bool):
        if train:
            return AlbumentationTransforms(self.build_train())
        else:
            return AlbumentationTransforms(self.build_test())

    @abstractmethod
    def build_train(self):
        pass

    @abstractmethod
    def build_test(self):
        pass


class AlbumentationTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        img = np.array(img)

        return self.transforms(image=img)['image']
