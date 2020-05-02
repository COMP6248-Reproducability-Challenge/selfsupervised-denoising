"""Configure the package logger.

Warning:
    This module has state and manipulates external use of the `logging` package.
"""

import sys
import os
import datetime
import logging

from colorlog import ColoredFormatter
from colored_traceback import Colorizer

FILE_FORMAT = "%(asctime)s %(name)-30s %(levelname)-8s %(message)s"
FILE_DATE_FORMAT = "%m-%d %H:%M:%S"
CONSOLE_FORMAT = "%(log_color)s%(message)s%(reset)s"


# Module level variable for tracking the console output handler
console_handle = None
# Package level logger
root_logger = logging.getLogger("")
package_logger = logging.getLogger(__name__.split(".")[0])
logger = logging.getLogger(__name__)


def _log_exception(exc_type, exc_value, exc_traceback):
    # The exception will still get thrown, so ensure it will not get written
    # to the console twice by removing the console handler before readding it
    if console_handle:
        root_logger.removeHandler(console_handle)
    if not issubclass(exc_type, KeyboardInterrupt):
        root_logger.error(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )
    if console_handle:
        root_logger.addHandler(console_handle)
    colorizer = Colorizer("default", False)
    sys.excepthook = colorizer.colorize_traceback
    colorizer.colorize_traceback(exc_type, exc_value, exc_traceback)


def setup(log_dir: str = None, filename: str = None):
    """Automatic logging setup for the SSDN package. Logging goes to
    an output file and the console. Console output is coloured using `colorlog` and
    any exception traceback is coloured using `coloured_traceback`. Handlers are
    attached to the root logger and will therefore capture and potentially manipulate output
    from any external packages.

    Args:
        log_dir (str, optional): Directory path to store logs in. Defaults to None.
            When None is used logs are not stored to a file.
        filename (str, optional): Filename to store log file as. Defaults to None.
            When None the current date and time will be used as the log name.
    """
    # Setup directory if it doesn't exist
    if log_dir is not None:
        if log_dir != "" and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # Use the current datetime if filename not set
        if filename is None:
            date = datetime.datetime.now()
            filename = "log_{date:%Y_%m_%d-%H_%M_%S}.txt".format(date=date)
        file_path = os.path.join(log_dir, filename)

        # Configure logging to a file
        file = logging.FileHandler(file_path, mode="w")
        formatter = logging.Formatter(fmt=FILE_FORMAT, datefmt=FILE_DATE_FORMAT)
        file.setLevel(logging.DEBUG)
        file.setFormatter(formatter)
        root_logger.addHandler(file)

    # Configure console logging
    # Must keep track of handle in global state so it can be removed before
    # uncaught exceptions are logged
    global console_handle  # noqa: E261
    if console_handle is None:
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        formatter = ColoredFormatter(CONSOLE_FORMAT, datefmt=None)
        console.setFormatter(formatter)
        root_logger.addHandler(console)
        console_handle = console
    # Attach hook to log any exceptions and add coloured traceback
    sys.excepthook = _log_exception
    # By default expose all messages by default
    package_logger.setLevel(logging.DEBUG)
