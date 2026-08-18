import whisperx
from ccgenerator.utils.low_resources import unload_model

def align(audio, speech_result, device, low_resources=False, **kwargs):
    load_keys = [
        "model_dir",
        "model_cache_only"
        ]
    model_a, metadata = whisperx.load_align_model(language_code=speech_result["language"], **kwargs[load_keys])
    align_keys = [
        "interpolate_method",
        "return_char_alignments",
        "print_progress"
        ]
    result = whisperx.align(
        speech_result["segments"],
        model_a,
        metadata,
        audio,
        device,
        kwargs[align_keys]
        )
    unload_model(model, low_resources=low_resources)

    return result
