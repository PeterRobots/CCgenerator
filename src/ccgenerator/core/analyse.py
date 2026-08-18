import math
import glob
from pathlib import Path
import pandas as pd
from pandas import DataFrame as DF
from collections import Counter

from utils.exceptions import InvalidFileTypeError
from typing import List


def write_to_srt():
    # SRT format prediction
    srt_start_time = datetime.timedelta(seconds = start_time)
    srt_end_time = datetime.timedelta(seconds = end_time)
    srt_prediction = srt.Subtitle(index=idx, start=srt_start_time, end=srt_end_time, content=f"The sound of {prediction['label']} in the background")
    srt_predictions.append(srt_prediction)
    # SRT
    srt_string = srt.compose(srt_predictions)
    with open(output_path/output_file.with_suffix(".srt"), 'w+') as f:
        f.write(srt_string)



def get_row_prediction_matches(data, row, p_set_label, row_p_set_label) -> List[int]:
    idx_match_list = data[data[p_set_label].apply(lambda x: x & getattr(row,row_p_set_label))].index.to_list()
    idx_match_list.pop(row.Index)

    return idx_match_list


def compare_segment_labels(data:DF) -> DF:
    for row in data.itertuples():
        # Reference point of row
        segments_within_row = data[data["end"] <= row.end or data["start"] >= row.start]
        data.at[row.Index, "p_matches"] = get_row_prediction_matches(data, row, "prediction_set", "prediction_set")
        data.at[row.Index, "p_matches_score"] = data["score"][data["p_matches"][row.Index]].mean()

    return data


def make_prediction_set(prediction:str) -> set:
    return set(prediction.lower().split(", "))


def adjust_row_times(saved_rows:DF, start:float, end:float) -> DF:
    row_list = []
    for row in saved_rows.itertuples():
        if row.end >= end:
            if row.start < start:
                row_copy = copy(row)
                row_copy.end = start
                row_list.append(row_copy)
            else:
                row.start = end
                row_list.append(row)
        elif row.start <= start:
                row.end = start
                row_list.append(row)
        else:
            row_list.append(row)
    return DF(row_list)


def get_primary_audio_in_segment(data, data_segment, score_threshold=0.1, skip_labels = ["speech"]) -> DF:
     # Go through in segment size order.
    saved_rows = DF(columns=data_segment.columns)
    # Ignore duplicated rows with multiple predictions.
    length_segment = len(data_segment["end"].drop_duplicates())
    for row in data_segment.itertuples():
        if any(label in row.label.lower() for label in skip_labels): # Skip if label contains any of skip labels
            continue
        if row.score > score_threshold:
            # Uncovered segment in final
            if row.end > saved_rows["end"].max() or row.start < saved_rows["start"].min():
                saved_rows = adjust_row_times(saved_rows, row.start, row.end)
                saved_rows = pd.concat([saved_rows, DF(row)])
            # 1) row score is higher than saved segments over same time window.
            elif row.score > saved_rows["score"][saved_rows["end"] >= row.end and saved_rows["start"] <= row.start].max:
                saved_rows = adjust_row_times(saved_rows, row.start, row.end)
                saved_rows = pd.concat([saved_rows, DF(row)])
            # 2) Segment label matches other all other segment labels and score is high enough
            elif len(row.p_matches) == length_segment:
                saved_rows = adjust_row_times(saved_rows, row.start, row.end)
                saved_rows = pd.concat([saved_rows, DF(row)])

    return saved_rows


def analyse(data, threshold_score=0.1) -> DF:
    if isinstance(data, Path):
        # load combined grouped segment data
        data = pd.read(data)
    elif not isinstance(data, DF):
        raise InvalidFileTypeError()
    analysed_output = {k:[] for k in data.columns.to_list()}
    # Make sets of the prediction strings
    data["prediction_set"] = data["prediction"].apply(make_prediction_set)
    print(data)
    # get segments
    segments = data.drop_duplicates(subset="segment").to_list()
    final_segments_list = []
    for s in segments:
        data_segment = data[data["segment"] == s]
        data_segment = compare_segment_labels(data_segment)
        final_segments_list.append(get_primary_audio_in_segment(data, s, score_threshold=score_threshold))
      # Look within segments, compare overlapping segments and pick highest score.ArithmeticError
      # Look at labelled value, if label matches for overlapping segment, pick largest segment or simply combine into new table entry for start -> end.
    # data["matching_segments"] =
    # segment_label_sets = []
    # for l in sement_labels:
    #     segment_label_sets.appends(set(l.lower().split(", ")))
    # DF(segment_label_sets, columns=[""])
    # shared_words = set1 & set2
    final_data = pd.concat(final_segments_list, ignore_index=True)

    return final_data
