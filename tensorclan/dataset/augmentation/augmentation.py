import tensorclan.dataset.augmentation as tc_augmentation
from tensorclan.utils import setup_logger

logger = setup_logger(__name__)

# a dict containing all the user defined transforms
_REGISTERED_AUGMENTATIONS = {}


def augmentation(cls):
    _REGISTERED_AUGMENTATIONS[cls.__name__] = cls
    logger.info(f'Registered {cls.__name__} as augmentation')
    return cls


def get_augmentation(ctor_name: str, *args, **kwargs):
    if hasattr(tc_augmentation, ctor_name):
        logger.info(f'Building {tc_augmentation.__name__}.{ctor_name}')
        return getattr(tc_augmentation, ctor_name)(*args, **kwargs)

    if ctor_name in _REGISTERED_AUGMENTATIONS:
        logger.info(f'Building User Augmentation {_REGISTERED_AUGMENTATIONS[ctor_name]}.{ctor_name}')
        return _REGISTERED_AUGMENTATIONS[ctor_name](*args, **kwargs)

    raise ModuleNotFoundError
