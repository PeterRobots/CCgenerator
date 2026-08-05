import whisperx
import gc
from whisperx.diarize import DiarizationPipeline
from whisperx.utils import get_writer
from pyannote.audio import Pipeline
from pyannote.audio.telemetry import set_telemetry_metrics
# disable metrics globally
set_telemetry_metrics(False, save_choice_as_default=True)


def speech(
    audio_file,
    model_name,
    model_cache_only,
    align_model_dir,
    align_model_cache_only,
    diarize_model,
    diarize_dir,
    device,
    compute_type,
    batch_size,
    HF_TOKEN,
    PYANNOTE_TOKEN = ""
    ):
    if not PYANNOTE_TOKEN:
        PYANNOTE_TOKEN = HF_TOKEN

    writer = get_writer(args["output_format"], ["output_dir"])
    writer_arg_list = ['highlight_words', 'max_line_width', 'max_line_count']
    writer_args = {k:args[k] for k in writer_arg_list}
    # Part 1: VAD & ASR Loop
    results = []
    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(model_name, device, compute_type=compute_type, download_root=model_name, use_auth_token=HF_TOKEN, local_files_only=model_cache_only)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    results.append((result, audio_file))
    print(result["segments"]) # before alignment

    # delete model if low on GPU resources
    # import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device, model_dir=align_model_dir, model_cache_only=align_model_cache_only)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    results.append((result, audio_file))
    print(result["segments"]) # after alignment

    # delete model if low on GPU resources
    # import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model_a


    results.append((result, audio_file))
    print(diarize_segments)
    print(result["segments"]) # segments are now assigned speaker IDs
    # >> Write
    for result, audio_path in results:
        result["language"] = 'en'
        writer(result, audio_path, writer_args)
