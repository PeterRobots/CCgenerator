import torch
from pathlib import Path
from configparser import ConfigParser
from dotenv import dotenv_values
from utils.parser import parse_arguments
from CCgenerator.utils.exceptions import MissingPyAnnoteToken, MissingHFToken, BadModelName

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
        if args['pyannote-model'] == 'precision' && args['pyannote-key'] is None:
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

    from CCgenerator.app import run
    run(args, parser)


if __name__ == "__main__":
    main()
