import whisperx
from ccgenerator.utils.low_resources import unload_model

def align(audio, speech_result, language_code, device, low_resources=False, **kwargs):
    model_a, metadata = whisperx.load_align_model(language_code=language_code, **kwargs)
    result = whisperx.align(speech_result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    unload_model(model, low_resources=low_resources)

    return result
