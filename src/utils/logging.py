import sys
import logging


def create_logger():
    """General logger"""
    log = logging.getLogger()
    logFormatter = logging.Formatter("[%(levelname)s] %(message)s")
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    if (log.hasHandlers()):
        log.handlers.clear()
    log.addHandler(consoleHandler)
    log.setLevel('INFO')
    return log
