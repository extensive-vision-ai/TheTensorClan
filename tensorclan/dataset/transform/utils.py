from tensorclan.utils import setup_logger

logger = setup_logger(__name__)

# a dict containing all the user defined transforms
_REGISTERED_TRANSFORMATIONS = {}


def transformation(cls):
    _REGISTERED_TRANSFORMATIONS[cls.__name__] = cls
    logger.info(f'Registered {cls.__name__} as transformation')
    return cls


def get_transforms(ctor_name: str, *args, **kwargs):
    import tensorclan.dataset.transform as tc_transform
    if hasattr(tc_transform, ctor_name):
        logger.info(f'Building {tc_transform.__name__}.{ctor_name}')
        return getattr(tc_transform, ctor_name)(*args, **kwargs)

    if ctor_name in _REGISTERED_TRANSFORMATIONS:
        logger.info(f'Building User transformation {_REGISTERED_TRANSFORMATIONS[ctor_name]}.{ctor_name}')
        return _REGISTERED_TRANSFORMATIONS[ctor_name](*args, **kwargs)

    raise ModuleNotFoundError
