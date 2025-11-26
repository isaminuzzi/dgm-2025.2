import logging
import os
from datetime import datetime
from pathlib import Path

import colorlog

def get_logger(
    name: str = "app",
    log_dir: str = "logs",
    level: str = "INFO",
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """
    Creates a configured logger that writes both to console and file.

    Args:
        name (str): Name of the logger (e.g., module name).
        log_dir (str): Directory to store log files.
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        console (bool): Whether to log to console.
        file (bool): Whether to log to a file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_level = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    logger = logging.getLogger(name)
    logger.setLevel(log_level[level])

    # Prevent duplicate handlers in notebooks or reloads
    if logger.handlers:
        return logger

    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Timestamped log file
    log_file = os.path.join(log_dir, f"{datetime.now():%Y%m%d}.log")

    # Formatters
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )

    # Console handler
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # File handler
    if file:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Optional: disable propagation to root logger
    logger.propagate = False

    return logger