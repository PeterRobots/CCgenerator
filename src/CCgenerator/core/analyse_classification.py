from pathlib import Path
import pandas as pd
from pandas import DataFrame as DF
import math
from typing import List
import glob
from collections import Counter


def get_row_prediction_matches(data, row, p_set_label, row_p_set_label) -> List[int]:
    idx_match_list = data[data[p_set_label].apply(lambda x: x & getattr(row,row_p_set_label))].index.to_list()
    idx_match_list.pop(row.Index)

    return idx_match_list


def compare_segment_labels(data:DF) -> DF:
    for row in data.itertuples():
        # Reference point of row
        segments_within_row = data[data["end"] <= row.end or data["start"] >= row.start]
        data.at[row.Index, "p_to_p_matches"] = get_row_prediction_matches(data, row, "prediction_set", "prediction_set")
        data.at[row.Index, "p2_to_p_matches"] = get_row_prediction_matches(data, row, "prediction_2_set", "prediction_set")
        data.at[row.Index, "p_to_p2_matches"] = get_row_prediction_matches(data, row, "prediction_set", "prediction_2_set")
        data.at[row.Index, "p2_to_p2_matches"] = get_row_prediction_matches(data, row, "prediction_2_set", "prediction_2_set")

    return data


def make_prediction_set(prediction:str) -> set:

    return set(prediction.lower().split(", ")


def analyse_segment_data(data) -> DF:
    # Make sets of the prediction strings
    data["prediction_set"] = data["prediction"].apply(make_prediction_set)
    data["prediction_2_set"] = data["prediction_2"].apply(make_prediction_set)
    print(data)
    # get segments
    segments = data.drop_duplicates(subset="segment").to_list()
    for s in segments:
        data_segment = data[data["segment"] == s]
        data_segment = compare_segment_labels(data)


    data["matching_segments"] =
    # segment_label_sets = []
    # for l in sement_labels:
    #     segment_label_sets.appends(set(l.lower().split(", ")))
    # DF(segment_label_sets, columns=[""])
    # shared_words = set1 & set2

    return data
