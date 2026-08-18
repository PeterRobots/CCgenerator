import argparse
import importlib.metadata
import platform
from whisperx.utils import (
    LANGUAGES,
    TO_LANGUAGE_CODE,
    optional_float,
    optional_int,
    str2bool
    )

def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(
        prog='CCgenerator',
        description='''
            This program generates Closed Captions for a supplied audio or video file.

            Configuration can be specified in a config.ini and/or via CLI.
            See `CCgenerator --help` for more information on CLI input.
        ''',
        usage='%(prog)s audio_file [options]'
        )
    parser.add_argument("--audio", "-i", nargs="+", type=str, help="audio file(s) to transcribe. If 'analyze' mode, supply data.csv instead")
    parser.add_argument("--mode", "-m", default=None, help="Mode of operation: all, speech, background, analyze.")
    parser.add_argument("--highlight-words", type=str2bool, default=None, help="Whether to highlight words.")
    parser.add_argument("--max-line-width", type=optional_int, default=None, help="Character line width to use for closed captions.")
    parser.add_argument("--max-line-count", type=optional_int, default=None, help="Number of lines for closed captions.")
    # whisperX
    parser.add_argument("--whisperx-model", type=str, default=None, help="name of the Whisper model to use")
    parser.add_argument("--whisperx-model-cache-only", type=str2bool, default=None, help="If True, will not attempt to download models, instead using cached models from --whisperx-model_dir")
    parser.add_argument("--whisperx-model-dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisperx by default")
    parser.add_argument("--whisperx-align-model-cache-only", type=str2bool, default=None, help="If True, will not attempt to download models, instead using cached models from --whisperx-align-model_dir")
    parser.add_argument("--whisperx-align-model-dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisperx by default")
    # Pyannote
    parser.add_argument("--pyannote-model", default=None, help="name of the PyAnnote model to use")
    parser.add_argument("--pyannote-model-dir", type=str, default=None, help="the path to save model files; uses ~/.cache/pyannote by default")
    parser.add_argument("--pyannote-key", type=str, default=None, help="PyAnnote key for access to premium precision model.")
    # ast
    parser.add_argument("--ast-model", type=str, default=None, help="name of the ast model to use")
    parser.add_argument("--ast-model-cache-only", type=str2bool, default=None, help="If True, will not attempt to download models, instead using cached models from --ast-model_dir")
    parser.add_argument("--ast-model-dir", type=str, default=None, help="the path to save model files; uses ~/.cache/ast by default")
    parser.add_argument("--score-threshold", type=optional_float, default=None, help="Minimum score threshold for background sound identification.")
    # Compute
    parser.add_argument("--low-resources", type=str2bool, default=None, help="Unload models from vram and ram after each step to save resources")
    parser.add_argument("--device", type=str, default=None, help="device type to use for PyTorch inference (e.g. cpu, cuda)")
    parser.add_argument("--device-index", type=optional_int, default=None, help="device index to use for FasterWhisper inference")
    parser.add_argument("--batch-size", type=optional_int, default=None, help="the preferred batch size for inference")
    parser.add_argument("--compute_type", type=str, default=None, choices=["default", "float16", "float32", "int8"], help="compute type for computation; 'default' uses float16 on GPU, float32 on CPU")
    parser.add_argument("--threads", type=optional_int, default=None, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="directory to save the outputs")
    parser.add_argument("--output-format", "-f", type=str, default=None, choices=["all", "srt", "vtt", "txt", "tsv", "json", "aud"], help="format of the output file; if not specified, all available formats will be produced")

    parser.add_argument("--verbose", type=str2bool, default=None, help="whether to print out the progress and debug messages")
    parser.add_argument("--log-level", type=str, default=None, choices=["debug", "info", "warning", "error", "critical"], help="logging level (overrides --verbose if set)")

    parser.add_argument("--config", "-c", type=str, default=None, help="Configuration (ini) file with settings for CCgenerator, whisperX, PyAnnote and ast")
    parser.add_argument("--secrets", "-s", type=str, default=None, help=".secrets file containing Write permission Hugging Face Access Token to access models")
    parser.add_argument("--hf-token", "-t", type=str, default=None, help="Write permission Hugging Face Access Token to access PyAnnote gated models")

    parser.add_argument("--print-progress", type=str2bool, default = None, help = "if True, progress will be printed.")
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {importlib.metadata.version('CCgenerator')}",help="Show CCgenerator version information and exit")
    parser.add_argument("--python-version", "-P", action="version", version=f"Python {platform.python_version()} ({platform.python_implementation()})",help="Show python version information and exit")
    args = {k:v for k,v in parser.parse_args().__dict__.items() if not v is None and not v == "None"}
    return args
