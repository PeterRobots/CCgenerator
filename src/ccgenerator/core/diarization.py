def diarize(audio, whisper_result, **kwargs):
     # 3. Assign speaker labels
    diarize_model = DiarizationPipeline(kwargs)
    diarize_segments = diarize_model(audio)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)
