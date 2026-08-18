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

from typing import Tuple

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


def get_temperatures(temperature, increment_temperature) -> Tuple[float]:
    temperature = args.pop("temperature")
    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = (temperature)

    return temperature


def main():
    """Run and parse cli input if available.

    Args:
        args: Dictionary of command-line arguments.
        parser: argparse.ArgumentParser object.
    """
    args = parse_arguments()
    args = {k.replace("-","_"):v for k,v in args.items()}

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

    if "hf_token" not in args:
        raise MissingHFToken()


    AST_MODELS = {
        "mit" : "ast-finetuned-audioset-10-10-04593",
        "qwen" : "Qwen/Qwen3-ASR-1.7B-hf"
        }
    arg_ast_model = args['ast_model'].lower()
    if arg_ast_model in AST_MODELS:
        args['ast_model'] = AST_MODELS[arg_ast_model]
    else:
        raise BadModelName("Missing ast model")


    audio_file: str = args.pop("audio")
    language = args.pop("language")
    log_level = args.pop("log_level")

    output_options = {
        "output_dir":args.pop("output_dir"),
        "output_format":args.pop("output_format"),
        "max_line_count":args.pop("max_line_count"),
        "max_line_width":args.pop("max_line_width"),
        "highlight_words":args.pop("highlight_words")
        }

    mode: str = args.pop("mode")
    # Compute
    compute_options = {
        "low_resources":args.pop("low_resources"),
        "device":args.pop("device"),
        "device_index":args.pop("device_index"),
        "batch_size":args.pop("batch_size"),
        "compute_type":args.pop("compute_type"),
        "threads":args.pop("threads")
        }

    temperatures = get_temperatures(args["temperature"], args["temperature_increment_on_fallback"])
    # ASR
    asr_options = {
        "beam_size": args.pop("beam_size"),
        "patience": args.pop("patience"),
        "length_penalty": args.pop("length_penalty"),
        "temperatures": temperature,
        "compression_ratio_threshold": args.pop("compression_ratio_threshold"),
        "log_prob_threshold": args.pop("logprob_threshold"),
        "no_speech_threshold": args.pop("no_speech_threshold"),
        "condition_on_previous_text": False,
        "initial_prompt": args.pop("initial_prompt"),
        "hotwords": args.pop("hotwords"),
        "suppress_tokens": [int(x) for x in args.pop("suppress_tokens").split(",")],
        "suppress_numerals": args.pop("suppress_numerals"),
    }
    asr_model_name: str = args.pop("asr_model")
    asr_model_dir: str = args.pop("asr_model_dir")
    asr_model_cache_only: bool = args.pop("asr_model_cache_only")
    align_model_dir: str = args.pop("align_model_dir")
    align_model_cache_only: bool = args.pop("align_model_cache_only")
    align_options = {
        "language": language,
    }

    # VAD
    VAD_MODELS = {
        "community" : "pyannote/speaker_diarization_community_1",
        "precision" :  "pyannote/speaker_diarization_precision_2",
        }

    vad_model_name = args.pop('vad_model').lower()

    if vad_model_name in VAD_MODELS:
        if vad_model_name == 'precision' and args['vad_key'] is None:
            raise MissingPyAnnoteToken
        else:
            vad_model_name = VAD_MODELS[vad_model_name]
    else:
        raise BadModelName("Missing VAD model")

    vad_model_dir: str = args.pop("vad_model_dir")
    vad_options={
        "chunk_size": args.pop("chunk_size"),
        "vad_onset": args.pop("vad_onset"),
        "vad_offset": args.pop("vad_offset"),
        "interpolate_method":args.pop("interpolate_method"),
        "return_char_alignments":args.pop("return_char_alignments"),
    },

    # AST
    ast_model_name: str = args.pop("ast_model")
    ast_model_dir: str = args.pop("ast_model_dir")
    ast_model_cache_only: bool = args.pop("ast_model_cache_only")

    # SECRETS
    HF_TOKEN: str = args.pop("hf_token")

    if "vad_key" in args:
        VAD_KEY:str = args.pop("vad_key")
    else:
        VAD_KEY:str = None

    # FLAGS
    VERBOSE: bool = args.pop("verbose")
    PRINT_PROGRESS: bool = args.pop("print_progress")

    logger.info("Loaded cli args.")
    # delete model if low on GPU resources
    # import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model
    results = []
    mode = mode.lower()
    if mode in ('speech', 'all'):
        speech_result = speech(
            audio_file,
            asr_model_name,
            asr_model_dir,
            asr_model_cache_only,
            HF_TOKEN,
            VAD_KEY,
            **compute_kwargs
            )

        results.append((speech_result, audio_file))

    if mode in ('speech', 'align', 'all'):
        result = align(
            audio_file,
            speech_result,
            device=compute_kwargs["device"],
            model_dir=align_model_dir,
            model_cache_only=align_model_cache_only,
            print_progress=print_progress,
            low_resources=compute_kwargs["low_resources"],
            **align_options
            )
        results.append((result, audio_file))

    if mode in ('speech', 'diarize', 'diarise', 'all'):
        result = diarize(
            audio_file,
            speech_result,
            model_name=diarize_model,
            token=VAD_TOKEN,
            device=compute_kwargs["device"],
            cache_dir=diarize_dir,
            low_resources=compute_kwargs["low_resources"],
            **vad_options
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
