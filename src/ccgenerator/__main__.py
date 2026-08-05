import os
import warnings
import torch
import gc
import torch
from pathlib import Path
from configparser import ConfigParser
from dotenv import dotenv_values
from utils.parser import parse_arguments
from ccgenerator import (
    align,
    background,
    speech,
    diarize,
    analyse
    )
from ccgenerator.utils.exceptions import MissingFileError, MissingHFToken, MissingPyAnnoteToken, BadModelName
from logger import get_logger


logger = get_logger(__name__)

def update_with_cli_args(args, cli_args) -> dict:
    args.update({k:v for k,v in cli_args.items if not v is None})

    return args


def get_secrets(args:dict) -> dict:
    secret_path = Path(args["secrets"])
    if secret_path.exists():
        secrets = dotenv_values(secret_path)
        secrets = {k.lower().replace("_","-"):v for k,v in secrets.items()}
        secrets = update_with_cli_args(secrets, args)

    return secrets


def get_config(args:dict) -> dict:
    # DEFAULT CONFIG
    default_config_args = load_config(".default_config.ini")
    # USER CONFIG
    if 'config' in args:
        config_path: Path = Path(args.pop("config"))
        config_args = load_config(config_path)
        default_config_args.update(config_args)
    # Update default config with cli args that are set.
    default_config_args = update_with_cli_args(default_config_args, args)

    return default_config_args


def load_config(config_path) -> dict:
    if config_path.is_dir():
        config = ConfigParser()
        config.read(config_path)
        logger.info("Loaded configuration file args.")
        args = config['CCgenerator'] | config['ASR'] | config['AST']
    else:
        logger.error("Invalid config file path.")


    return args


def main():
    """Run and parse cli input if available.

    Args:
        args: Dictionary of command-line arguments.
        parser: argparse.ArgumentParser object.
    """
    args = parse_arguments()

    log_level = args.get("log_level")
    verbose = args.get("verbose")

    if log_level is not None:
        setup_logging(level=log_level)
    elif verbose:
        setup_logging(level="info")
    else:
        setup_logging(level="warning")

    # Get config and update args.
    args = get_config(args)

    # Get secrets, should be HF_TOKEN and optionally PYANNOTE_TOKEN
    args = get_secrets(args)

    if args['device'] == 'default':
        args['device'] = 'cuda' if torch.cuda.is_available() else "cpu"

    if "hf-token" not in args:
        raise MissingHFToken()

    PYANNOTE_MODELS = {
        "community" : "pyannote/speaker-diarization-community-1",
        "precision" :  "pyannote/speaker-diarization-precision-2"
        }

    args['pyannote-model'] = args['pyannote-model'].lower()

    if args['pyannote-model'] in PYANNOTE_MODELS:
        if args['pyannote-model'] == 'precision' and args['pyannote-key'] is None:
            raise MissingPyAnnoteToken
        else:
            args['pyannote-model'] = PYANNOTE_MODELS[args['pyannote-model']]
    else:
        raise BadModelName()

    AST_MODELS = {
        "mit" : "ast-finetuned-audioset-10-10-04593",
        "qwen" : "Qwen/Qwen3-ASR-1.7B-hf"
        }

    if args['ast-model'].lower() in PYANNOTE_MODELS:
        args['ast-model'] = AST_MODELS[args['pyannote-model'].lower()]
    else:
        raise BadModelName()


    audio_file: str = args.pop("audio")
    mode: str = args.pop("mode")
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
    # delete model if low on GPU resources
    # import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model

    match mode.lower():
        case 'all':
            results = []

            result = speech(model_name, device=device, compute_type=compute_type, download_root=whisperx_model_name, use_auth_token=HF_TOKEN, local_files_only=whisperx_model_cache_only)
            results.append((result, audio_file))

            if save_resources:
                gc.collect(); torch.cuda.empty_cache(); del model

            result = align(audio, device=device, model_dir=whisperx_align_model_dir, model_cache_only=whisperx_align_model_cache_only)
            results.append((result, audio_file))

            if save_resources:
                gc.collect(); torch.cuda.empty_cache(); del model

            result = diarize(model_name=diarize_model, token=PYANNOTE_TOKEN, device=device, cache_dir=diarize_dir)
            results.append((result, audio_file))
            write_results(results, output_format=output_format, highlight_words=highlight_words, max_line_count=max_line_count, max_line_width=max_line_width)

        case 'speech':
            results = []

            result = speech(model_name, device=device, compute_type=compute_type, download_root=whisperx_model_name, use_auth_token=HF_TOKEN, local_files_only=whisperx_model_cache_only)
            results.append((result, audio_file))

            if save_resources:
                gc.collect(); torch.cuda.empty_cache(); del model

            result = align(audio, device=device, model_dir=whisperx_align_model_dir, model_cache_only=whisperx_align_model_cache_only)
            results.append((result, audio_file))

            if save_resources:
                gc.collect(); torch.cuda.empty_cache(); del model

            result = diarize(audio_file, whisper_result, model_name=diarize_model, token=PYANNOTE_TOKEN, device=device, cache_dir=diarize_dir)
            results.append((result, audio_file))
            write_results(results, output_format=output_format, highlight_words=highlight_words, max_line_count=max_line_count, max_line_width=max_line_width)

        case 'background':
            result = background(audio_file)
            analyse(result)
        case 'analyse' | 'analyze':
            analyse(audio_file)



if __name__ == "__main__":
    main()
