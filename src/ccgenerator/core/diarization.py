from ccgenerator.utils.low_resources import unload_model
from whisperx import assign_word_speakers

def diarize(audio, speech_result, low_resources=False, **kwargs):
     # 3. Assign speaker labels
     diarize_model = DiarizationPipeline(kwargs)
     diarize_segments = diarize_model(audio)
     # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
     result = assign_word_speakers(diarize_segments, speech_result)
     unload_model(diarize_model, low_resources=low_resources)

     return result
