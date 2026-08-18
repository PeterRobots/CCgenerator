import importlib
import importlib.metadata

__version__ = importlib.metadata.version("ccgenerator")

from .core.alignment import align
from .core.background import background
from .core.speech import speech
from .core.diarization import diarize
from .core.analyse import analyse


def _lazy_import(name):
    module = importlib.import_module(f"ccgenerator.{name}")
    return module


def setup_logging(*args, **kwargs):
    """
    Configure logging for ccgenerator.

    Args:
        level: Logging level (debug, info, warning, error, critical). Default: warning
        log_file: Optional path to log file. If None, logs only to console.
    """
    logging_module = _lazy_import("utils.logger")
    return logging_module.setup_logging(*args, **kwargs)


def get_logger(*args, **kwargs):
    """
    Get a logger instance for the given module.

    Args:
        name: Logger name (typically __name__ from calling module)

    Returns:
        Logger instance configured with ccgenerator settings
    """
    logging_module = _lazy_import("utils.logger")
    return logging_module.get_logger(*args, **kwargs)
