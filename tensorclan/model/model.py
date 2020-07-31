from tensorclan.utils import setup_logger

logger = setup_logger(__name__)

# a dict containing all the user defined models
_REGISTERED_MODEL = {}


def model(cls):
    _REGISTERED_MODEL[cls.__name__] = cls
    logger.info(f'Registered {cls.__name__} as model')
    return cls


def get_model(ctor_name: str, *args, **kwargs):
    import tensorclan.model.zoo as tc_model
    if hasattr(tc_model, ctor_name):
        logger.info(f'Building {tc_model.__name__}.{ctor_name}')
        return getattr(tc_model, ctor_name)(*args, **kwargs)

    if ctor_name in _REGISTERED_MODEL:
        logger.info(f'Building User Model {_REGISTERED_MODEL[ctor_name]}.{ctor_name}')
        return _REGISTERED_MODEL[ctor_name](*args, **kwargs)

    raise ModuleNotFoundError
