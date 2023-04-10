import logging
import sys
import os

# log level
level_dict = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'crit': logging.CRITICAL
}

LOG_PATH = "run.log"
LOG_FMT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
DATE_FMT = "%m/%d/%Y %H:%M:%S %p"
LOG_LEVEL = "debug"
# ENABLE_LOG = true
TO_CONSOLE = True
TO_FILE = False

loggers = dict()


def get_logger(name=None):
    global loggers

    if name is None:
        name = __name__

    if loggers.get(name):
        return loggers.get(name)

    fmt = logging.Formatter(fmt=LOG_FMT,
                            datefmt=DATE_FMT)
    logger = logging.getLogger(LOG_PATH)
    logger.setLevel(level_dict[LOG_LEVEL])

    if TO_CONSOLE:
        console_handler = logging.StreamHandler(sys.stdout)  # 默认是sys.stderr
        console_handler.setLevel(level_dict[LOG_LEVEL])
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

    if TO_FILE:
        log_dir = os.path.dirname(LOG_PATH)
        if os.path.isdir(log_dir) and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(LOG_PATH, encoding='utf-8')
        file_handler.setLevel(level_dict[LOG_LEVEL])
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    loggers[name] = logger
    return logger


logger = get_logger()
