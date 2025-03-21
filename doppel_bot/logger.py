import logging
import sys

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
