import importlib
import importlib.metadata

__version__ = importlib.metadata.version("ccgenerator")

from .core.alignment import align
from .core.background import background_classifier
from .core.speech import speech_classifier
from .core.diarisation import diarisation_classifier
from .core.analyse import analyser

def _lazy_import(name):
    module = importlib.import_module(f"audioanalyzer.{name}")
    return module


def setup_logging(*args, **kwargs):
    """
    Configure logging for audioanalyzer.

    Args:
        level: Logging level (debug, info, warning, error, critical). Default: warning
        log_file: Optional path to log file. If None, logs only to console.
    """
    logging_module = _lazy_import("logger")
    return logging_module.setup_logging(*args, **kwargs)


def get_logger(*args, **kwargs):
    """
    Get a logger instance for the given module.

    Args:
        name: Logger name (typically __name__ from calling module)

    Returns:
        Logger instance configured with audioanalyzer settings
    """
    logging_module = _lazy_import("logger")
    return logging_module.get_logger(*args, **kwargs)
