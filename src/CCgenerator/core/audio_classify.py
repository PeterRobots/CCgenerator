from transformers import AutoModelForAudioClassification, AutoConfig, pipeline, AutoFeatureExtractor, AutoTokenizer
from datasets import Dataset, Audio, ClassLabel, Features
from peft import PeftModel
import torch
import librosa
import srt
import datetime
from pathlib import Path
import pandas as pd
from pandas import DataFrame as df
import math
from typing import List
import glob

def audio_classifier():

    # Load base model configuration and set for FSD50K multi-label task
    base_config = AutoConfig.from_pretrained("ast-finetuned-audioset-10-10-04593")
    # base_config.num_labels = 200
    base_config.problem_type = "multi_label_classification"

    # Load base model (87M parameters, ~340 MB)
    model = AutoModelForAudioClassification.from_pretrained(
        "ast-finetuned-audioset-10-10-04593",
        config=base_config,
        ignore_mismatched_sizes=True
    )

    # # Apply LoRA adapter
    # model = PeftModel.from_pretrained(base_model, "audioforge-ast-fsd50k")
    # model.eval()
    # logits = model(input_values=input_values).logits
    # probs = torch.sigmoid(logits)  # Multi-label probabilities for each class
    # tokenizer = AutoTokenizer.from_pretrained("ast-finetuned-audioset-10-10-04593")
 ### Without PIPELINE
    feature_extractor = AutoFeatureExtractor.from_pretrained("ast-finetuned-audioset-10-10-04593")
#     inputs = feature_extractor(audio_file, sampling_rate=sampling_rate, return_tensors="pt")
#
#     with torch.no_grad():
#         logits = model(**inputs).logits
#     predicted_class_ids = torch.argmax(logits).item()
#     predicted_label = model.config.id2label[predicted_class_ids]
#     print(predicted_label)

    ### PIPELINE
    classifier = pipeline("audio-classification", model=model, feature_extractor=feature_extractor)
    # classifier = pipeline("audio-classification", model="./ast-finetuned-audioset-10-10-04593/")
    # classifier.load_lora_weights(
    #     "./audioforge-ast-fsd50k",
    #     weight_name="adapter_model.safetensors",
    #     # adapter_name="cereal"
    # )
    # result = classifier(audio_file)
    # for i,r in enumerate(result):
    #     label_id = int(r['label'].replace('LABEL_',''))
    #     result[i]['label'] = base_config.id2label[label_id]
    #     print(label_id)
    #     print(base_config.id2label[label_id])
    # print(result)
    return classifier


def load_audio(audio_path:Path, sample_rate=16000):
    # Load audio file (automatically resampled to 16kHz for AST)
    sounds, sample_rate = librosa.load(audio_path, sr=sample_rate)
    return sounds, sample_rate


def classify_segment(classifier, sounds, segment_length=2, segment_margin=0.0, sample_rate=16000, output_path=Path('')):
    if math.isclose(segment_margin, 0.0, abs_tol=1e-09):
        segment_margin = segment_length / 4.0
    # Define segmentation parameters
    samples_per_segment = segment_length * sample_rate
    samples_per_margin = round(segment_margin * sample_rate)
    # print(samples_per_segment, samples_per_margin)
    prediction_strings = []
    srt_predictions = []
    segment_dict = {
        "index":[],
        "start":[],
        "end":[],
        "prediction":[],
        "score":[],
        "prediction_2":[],
        "score_2":[],
        }
    max_time = len(sounds)
    increment = samples_per_segment - samples_per_margin
    # Process chunks sequentially
    for idx, start in enumerate(range(samples_per_margin, max_time + samples_per_margin, increment)):
        sample_start = start - samples_per_margin
        # print(start, sample_start)
        sample_end = sample_start + samples_per_segment
        if sample_end >= max_time:
            sample_end = max_time - 1

        chunk = sounds[sample_start : sample_end]

        # Skip trailing chunks that are too short to classify accurately
        if len(chunk) < sample_rate:
            continue

        # Calculate timestamps
        start_time = sample_start / sample_rate
        end_time = sample_end / sample_rate

        # Run inference on the audio segment
        predictions = classifier(chunk)
        # skip if top prediction is speech and second is low score.
        if predictions[0]["label"].lower() == 'speech':
            if predictions[1]['score'] < 0.1:
                continue
            else:
                predictions[0] = predictions[1]
                predictions[1] = predictions[2]
        # Skip if top prediction is too low
        elif predictions[0]["score"] < 0.1:
            continue
        top_prediction = predictions[0]
        prediction_string = f"[{start_time:.1f}s - {end_time:.1f}s] {top_prediction['label']}: {top_prediction['score']:.2%}"
        prediction_strings.append(prediction_string)
        # SRT format prediction
        srt_start_time = datetime.timedelta(seconds = start_time)
        srt_end_time = datetime.timedelta(seconds = end_time)
        srt_prediction = srt.Subtitle(index=idx, start=srt_start_time, end=srt_end_time, content=f"The sound of {top_prediction['label']} in the background")
        srt_predictions.append(srt_prediction)
        # CSV
        segment_dict["index"].append(idx)
        segment_dict["start"].append(start_time)
        segment_dict["end"].append(end_time)
        segment_dict["prediction"].append(predictions[0]['label'])
        segment_dict["score"].append(predictions[0]['score'])
        segment_dict["prediction_2"].append(predictions[1]['label'])
        segment_dict["score_2"].append(predictions[1]['score'])
# index (int or None) – The SRT index for this subtitle
# start (datetime.timedelta) – The time that the subtitle should start being shown
# end (datetime.timedelta) – The time that the subtitle should stop being shown
# proprietary (str) – Proprietary metadata for this subtitle
# content (str) – The subtitle content. Should not contain OS-specific line separators, only \n. This is taken care of already if you use srt.parse() to generate Subtitle objects.
    output_file = Path(f"audio_described_{samples_per_segment}_{samples_per_margin}")
    with open(output_path/output_file.with_suffix(".txt"), 'w+') as f:
        f.writelines(prediction_strings)
    # CSV
    # print(segment_dict)
    data = df(segment_dict)
    data.to_csv(output_path/output_file.with_suffix(".csv"), index=False)
    # SRT
    srt_string = srt.compose(srt_predictions)
    with open(output_path/output_file.with_suffix(".srt"), 'w+') as f:
        f.write(srt_string)


def group_segments(data:df, output_path:Path):
    data["width"] =  data["end"] - data["start"]
    max_segment_size = round(data["width"].max())
    last_time = round(data["end"].max())
    print(f"max_segment_size = {max_segment_size}")
    print(f"last_time = {last_time}")
    segment_bins = range(0, last_time + max_segment_size, max_segment_size)
    # print(segment_bins, len(segment_bins))
    print(f"segment_bins = {segment_bins}")
    labels = range(0, len(segment_bins) - 1)
    print(f"labels = {labels}")
    # New column for segment bin.
    data["segment"] = pd.cut(data["end"], bins=segment_bins, labels=labels)
    print(data["segment"])
    data.to_csv(output_path)
    return data


def load_glob_csv_data(data_path_pattern:Path):
    files = glob.glob(data_path_pattern.as_posix())
    main_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True, axis=0)
    sorted_df = main_df.sort_values(by='start')
    return sorted_df


def sort_audio_segment_dataframes(list_of_data:List[df]):
    # sort list of data dicts by resolution, large segment -> small.
    segments = {}
    for i, data in enumerate(list_of_data):
        segment_size = data[0].end - data[0].start
        segments[i] = segment_size

    sorted_segments = dict(sorted(segments.items(), key=lambda item: item[1], reverse=True))
    list_of_data = [list_of_data[i] for i in list(sorted_segments.keys())]


def make_combined_segment_data(data_path):
    data_path_pattern = data_path / "*.csv"
    output_path = data_path / "all_segments.csv"
    data = load_csv_data(data_path_pattern)
    compare_prediction_resolutions(data, output_path)

