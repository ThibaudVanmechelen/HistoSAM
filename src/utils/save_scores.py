import json
import numpy as np

def save_scores(scores : dict, scores_file : str, averages_file : str):
    json_scores = {key: [float(value) for value in values] for key, values in scores.items()}

    with open(scores_file, 'w') as f:
        json.dump(json_scores, f, indent = 4)

    averages = {key: float(np.mean(values)) for key, values in scores.items()}

    with open(averages_file, 'w') as f:
        json.dump(averages, f, indent = 4)