from tensorclan.utils import setup_logger

logger = setup_logger(__name__)

# a dict containing all the user defined dataset
_REGISTERED_DATASET = {}


def dataset(cls):
    _REGISTERED_DATASET[cls.__name__] = cls
    logger.info(f'Registered {cls.__name__} as dataset')
    return cls


def get_dataset(ctor_name: str, *args, **kwargs):
    return get_dataset_cls(ctor_name=ctor_name)(*args, **kwargs)


def get_dataset_cls(ctor_name: str):
    import tensorclan.dataset.zoo as tc_dataset
    if hasattr(tc_dataset, ctor_name):
        logger.info(f'Building {tc_dataset.__name__}.{ctor_name}')
        return getattr(tc_dataset, ctor_name)

    if ctor_name in _REGISTERED_DATASET:
        logger.info(f'Building User Dataset {_REGISTERED_DATASET[ctor_name]}.{ctor_name}')
        return _REGISTERED_DATASET[ctor_name]

    raise ModuleNotFoundError
