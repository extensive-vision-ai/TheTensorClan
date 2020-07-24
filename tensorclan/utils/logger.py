import logging
import sys
from logging import StreamHandler, Logger, Formatter

LOG_LEVEL = logging.INFO


def setup_logger(name: str) -> Logger:
    r"""
    sets up a `Logger` instance for logging to `stdout`

    Args:
        name: name of the current python module i.e. __name__

    Returns:
        Logger: the logger that can be used to log stuff

    """
    logger: Logger = logging.getLogger(f'{name}')

    if not logger.hasHandlers():
        logger.setLevel(LOG_LEVEL)  # set the logging level
        # logging format
        logger_format: Formatter = Formatter(
            '[ %(asctime)s - %(name)s ] %(levelname)s: %(message)s'
        )

        stream_handler: StreamHandler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logger_format)
        logger.addHandler(stream_handler)
        logger.propagate = False

    return logger  # return the logger
