import whisperx
import gc
from whisperx.diarize import DiarizationPipeline
from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE
from ccgenerator.utils.low_resources import unload_model
from pyannote.audio import Pipeline
from pyannote.audio.telemetry import set_telemetry_metrics
# disable metrics globally
set_telemetry_metrics(False, save_choice_as_default=True)


def check_language(language:str, model_name) -> str:
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")
    if model_name.endswith(".en") and language != "en":
        if language is not None:
            warnings.warn(
                f"{model_name} is an English-only model but received '{language}'; using English instead."
            )
        language = "en"

    return language


def speech(
    audio_file,
    language,
    model_name,
    model_dir,
    model_cache_only,
    HF_TOKEN,
    PYANNOTE_TOKEN,
    low_resources=False,
    **kwargs
    ):
    if not PYANNOTE_TOKEN:
        PYANNOTE_TOKEN = HF_TOKEN

    language = check_language(language, model_name)
    # Part 1: VAD & ASR Loop
    results = []
    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(
        model_name,
        download_root=model_dir,
        use_auth_token=HF_TOKEN,
        local_files_only=model_cache_only,
        device=kwargs["device"],
        compute_type=kwargs["compute_type"]
        )

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=int(kwargs["batch_size"]))
    unload_model(model, low_resources=low_resources)

    return result
