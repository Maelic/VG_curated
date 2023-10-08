import logging
import os
import sys

current_step = 0
use_step = False

def setup_logger(name, verbose=False):
    global use_step
    try:
        from loguru import logger
        level = "DEBUG" if verbose else "INFO"
        logger.remove()

        logger.add(sys.stdout, format='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>', colorize=True, level=level)
        logger.info("Using loguru logger with level: "+level)
        return logger
      
    except ImportError:
        level = logging.DEBUG if verbose else logging.INFO
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.info("Using default logger")
        # don't log results for the non-master process

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
