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
    device,
    compute_type,
    batch_size,
    HF_TOKEN,
    PYANNOTE_TOKEN,
    low_resources=False
    ):
    if not PYANNOTE_TOKEN:
        PYANNOTE_TOKEN = HF_TOKEN
    # Part 1: VAD & ASR Loop
    results = []
    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(model_name, device, compute_type=compute_type, download_root=model_dir, use_auth_token=HF_TOKEN, local_files_only=model_cache_only)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    unload_model(model, low_resources=low_resources)

    return result
