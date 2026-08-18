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
    analyse,
    get_logger,
    setup_logging
    )
from ccgenerator.utils.exceptions import MissingFileError, MissingHFToken, MissingPyAnnoteToken, BadModelName
from ccgenerator.utils.writer import write_results

logger = get_logger(__name__)


def update_with_cli_args(args, cli_args) -> dict:
    args.update({k:v for k,v in cli_args.items() if not v is None})

    return args


def get_secrets(args:dict) -> dict:
    secret_path = Path(args["secrets"])
    if secret_path.exists():
        secrets = dotenv_values(secret_path)
        secrets = {k.lower().replace("_","-"):v for k,v in secrets.items()}
        secrets = update_with_cli_args(secrets, args)

    return secrets


def load_config(config_path:Path) -> dict:
    if config_path.is_file():
        config = ConfigParser()
        config.read(config_path)
        logger.info("Loaded configuration file args.")
        args = {}
        for section in config.sections():
            args.update({k:v for k,v in config.items(section) if not v is None and not v == "None"})
        # args = dict(config.get('CCgenerator')) | dict(config.get('ASR')) | dict(config.get('AST'))
    else:
        logger.error("Invalid config file path.")
        raise MissingFileError("Invalid config file path.")

    return args


def get_config(args:dict) -> dict:
    # DEFAULT CONFIG
    import importlib.resources as resources
    package_path = resources.files("ccgenerator")
    default_config_path = package_path / ".default_config.ini"
    default_config_args = load_config(default_config_path)
    # USER CONFIG
    if 'config' in args:
        config_path: Path = Path(args.pop("config"))
        config_args = load_config(config_path)
        default_config_args.update(config_args)
    # Update default config with cli args that are set.
    default_config_args = update_with_cli_args(default_config_args, args)

    return default_config_args


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
    if "secrets" in args:
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
        raise BadModelName("Missing Pyannote model")

    AST_MODELS = {
        "mit" : "ast-finetuned-audioset-10-10-04593",
        "qwen" : "Qwen/Qwen3-ASR-1.7B-hf"
        }
    arg_ast_model = args['ast-model'].lower()
    if arg_ast_model in AST_MODELS:
        args['ast-model'] = AST_MODELS[arg_ast_model]
    else:
        raise BadModelName("Missing ast model")


    audio_file: str = args.pop("audio")
    output_kwargs_keys = [
        "output-dir",
        "output-format",
        "max-line-count",
        "max-line-width",
        "highlight-words"
        ]
    output_kwargs = {}
    for k in output_kwargs_keys:
        if k in args:
            output_kwargs.update({k.replace("-","_"):args[k]})

    mode: str = args.pop("mode")
    # Compute
    compute_kwargs_keys = [
        "low-resources",
        "device",
        "device-index",
        "batch-size",
        "compute-type",
        "log-level",
        "threads"
        ]
    compute_kwargs = {}
    for k in compute_kwargs_keys:
        if k in args:
            compute_kwargs.update({k.replace("-","_"):args[k]})

    # whisperX
    whisperx_model_name: str = args.pop("whisperx-model")
    whisperx_model_dir: str = args.pop("whisperx-model-dir")
    whisperx_model_cache_only: bool = args.pop("whisperx-model-cache-only")
    whisperx_align_model_dir: str = args.pop("whisperx-align-model-dir")
    whisperx_align_model_cache_only: bool = args.pop("whisperx-align-model-cache-only")

    # PyAnnote
    pyannote_model_name: str = args.pop("pyannote-model")
    pyannote_model_dir: str = args.pop("pyannote-model-dir")

    # AST
    ast_model_name: str = args.pop("ast-model")
    ast_model_dir: str = args.pop("ast-model-dir")
    ast_model_cache_only: bool = args.pop("ast-model-cache-only")

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
    results = []
    mode = mode.lower()
    if mode in ('speech', 'all'):
        speech_result = speech(
            audio_file,
            whisperx_model_name,
            whisperx_model_dir,
            whisperx_model_cache_only,
            HF_TOKEN,
            PYANNOTE_KEY,
            **compute_kwargs
            )

        results.append((speech_result, audio_file))

    if mode in ('speech', 'align', 'all'):
        result = align(
            audio_file,
            speech_result,
            device=compute_kwargs["device"],
            model_dir=whisperx_align_model_dir,
            model_cache_only=whisperx_align_model_cache_only,
            batch_size=compute_kwargs["batch_size"],
            low_resources=compute_kwargs["low_resources"]
            )
        results.append((result, audio_file))

    if mode in ('speech', 'diarize', 'diarise', 'all'):
        result = diarize(
            audio_file,
            speech_result,
            model_name=diarize_model,
            token=PYANNOTE_TOKEN,
            device=compute_kwargs["device"],
            cache_dir=diarize_dir,
            low_resources=compute_kwargs["low_resources"]
            )
        results.append((result, audio_file))

    if mode in ('background', 'all'):
        result = background(
            audio_file,
            ast_model_name,
            model_path=ast_model_dir,
            cache_only=ast_model_cache_only,
            device=compute_kwargs["device"],
            output_path=output_kwargs["output_dir"],
            low_resources=compute_kwargs["low_resources"]
            )
        results.append((result, audio_file))

    if mode in ('analyse', 'analyze', 'all'):
        analyse(audio_file, results)

    write_results(results, **output_kwargs_keys)



if __name__ == "__main__":
    main()
