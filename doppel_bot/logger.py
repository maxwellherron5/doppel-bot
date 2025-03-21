import functools
import logging
import sys
import time
import typing
from typing import TypeVar

T = TypeVar("T")
_DEFAULT_LOG_LEVEL = logging.INFO


def get_logger(name="doppel-bot") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(_DEFAULT_LOG_LEVEL)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(_DEFAULT_LOG_LEVEL)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def log_duration(f: typing.Callable[..., T]):
    @functools.wraps(f)
    def wrapper(*args, **kwargs) -> T:
        _logger = get_logger(__name__)
        start = time.time_ns() / (1000 * 1000)
        _logger.info(f"start {f.__name__!r}")
        val = f(*args, **kwargs)
        end = time.time_ns() / (1000 * 1000)
        _logger.info(f"end {f.__name__!r} duration: {end - start} ms")
        return val

    return wrapper
