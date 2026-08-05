import os
import warnings
from ccgenerator.utils.exceptions import MissingFileError, MissingHFToken, MissingPyAnnoteToken
from ccgenerator.speech import run_whisperx
from logger import get_logger

logger = get_logger(__name__)


def run(args: dict, parser: argparse.ArgumentParser):
    """Run and parse cli input if available.

    Args:
        args: Dictionary of command-line arguments.
        parser: argparse.ArgumentParser object.
    """

    audio_file: str = args.pop("audio")
    # Compute
    low_resources = args.pop("low-resources")
    device: str = args.pop("device")
    output_dir: str = args.pop("output-dir")
    output_format: str = args.pop("output-format")
    device_index: int = args.pop("device-index")
    batch_size: int = args.pop("batch-size")
    compute_type: str = args.pop("compute-type")
    log_level: str = args.pop("log-level")
    cpu_threads: int = args.pop("threads")

    # whisperX
    whisperx_model_name: str = args.pop("whisperx-model")
    whisperx_model_dir: str = args.pop("whisperx-model-dir")
    whisperx_model_cache_only: bool = args.pop("whisperx-model-cache-only")
    whisperx_align_model_dir: str = args.pop("whisperx-align-model-dir")
    whisperx_align_model_cache_only: bool = args.pop("whisperx-align-model-cache-only")

    # PyAnnote
    pyannote_model_name: str = args.pop("pyannote-model")
    pyannote_model_dir: str = args.pop("pyannote-model-dir")

    # ASP
    asp_model_name: str = args.pop("asp-model")
    asp_model_dir: str = args.pop("asp-model-dir")
    asp_model_cache_only: bool = args.pop("asp-model-cache-only")

    # SECRETS
    HF_TOKEN: str = args.pop("hf-token")

    if "pyannote-key" in args:
        PYANNOTE_KEY:str = args.pop("pyannote-key")
    else:
        PYANNOTE_KEY:str = None

    # FLAGS
    VERBOSE: bool = args.pop("verbose")
    PRINT_PROGRESS: bool = args.pop("print-progress")

    logger.info("Loaded cli args.")

    run_whisperx(
        audio_file,
        model_name,
        model_cache_only,
        align_model_dir,
        align_model_cache_only,
        pyannote_model_name,
        pyannote_model_dir,
        device,
        compute_type,
        batch_size,
        HF_TOKEN,
        PYANNOTE_KEY=PYANNOTE_KEY
        )

if __name__ == "__main__":
    run()
