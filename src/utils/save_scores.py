"""File for the save_scores util function."""

import json
import numpy as np

def save_scores(scores : dict, scores_file : str, averages_file : str):
    """
    Function to save the scores.

    Args:
        scores (dict): the scores given as a dict.
        scores_file (str): the path of the file where to save the raw score arrays.
        averages_file (str): the path of the file where to save the averaged scores.
    """
    json_scores = {key: [float(value) for value in values] for key, values in scores.items()}

    with open(scores_file, 'w') as f:
        json.dump(json_scores, f, indent = 4)

    averages = {key: float(np.mean(values)) for key, values in scores.items()}

    with open(averages_file, 'w') as f:
        json.dump(averages, f, indent = 4)