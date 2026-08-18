import whisperx
import gc
from whisperx.diarize import DiarizationPipeline
from ccgenerator.utils.low_resources import unload_model
from pyannote.audio import Pipeline
from pyannote.audio.telemetry import set_telemetry_metrics
# disable metrics globally
set_telemetry_metrics(False, save_choice_as_default=True)


def speech(
    audio_file,
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
    # Part 1: VAD & ASR Loop
    results = []
    # 1. Transcribe with original whisper (batched)
    print("speech input: \n",
        audio_file,
        model_name,
        model_dir,
        model_cache_only,
        HF_TOKEN,
        PYANNOTE_TOKEN
        )
    model = whisperx.load_model(
        model_name,
        download_root=model_dir,
        use_auth_token=HF_TOKEN,
        local_files_only=model_cache_only,
        device=kwargs["device"],
        compute_type=kwargs["compute_type"]
        )

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=kwargs["batch_size"])
    unload_model(model, low_resources=low_resources)

    return result
