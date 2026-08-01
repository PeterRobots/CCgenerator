from CCgenerator.core.audio_classify import load_audio, audio_classifier, classify_segment, load_glob_csv_data, group_segments


def run_asp(audio_file, data_path):
    project_path = audio_file.parent
    data_path = data_path / "cc_data"
    if not data_path.is_dir():
        data_path = audio_path.parent
    sounds, sample_rate = load_audio(audio_file)
    segment_lengths = [20, 10, 5, 2]
    classifier = audio_classifier()
    for sl in segment_lengths:
        classify_segment(classifier, sounds, segment_length=sl, sample_rate=sample_rate, output_path=data_path)
    # Load and group the segment data
    data_path_pattern = data_path / "*.csv"
    output_path = data_path / "all_segments.csv"
    data = load_glob_csv_data(data_path_pattern)
    group_segments(data, output_path)
    # load combined grouped segment data
    data = pd.read(data_path / "all_segments.csv")
    analyse_segment_data(data)
