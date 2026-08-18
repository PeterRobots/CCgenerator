from whisperx.utils import get_writer

def write_results(
        results,
        output_dir,
        output_format='srt',
        highlight_words=False,
        max_line_count=2,
        max_line_width=33,
        **kwargs
        ) -> None:
    writer = get_writer(output_format, output_dir)
    for result, audio_path in results:
        result["language"] = 'en'
        writer(
            result,
            audio_path,
            highlight_words=highlight_words,
            max_line_count=max_line_count,
            max_line_width=max_line_width,
            **kwargs
            )
