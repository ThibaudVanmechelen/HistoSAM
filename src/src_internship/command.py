'''Evaluate the model with different configurations'''
import subprocess

import torch
from evaluate import evaluate_with_config
from utils.config import load_config


def run_command(command):
    '''Run command in windows powers shell'''
    process = subprocess.Popen(command, shell=True)
    out, err = process.communicate()
    return out, err

if __name__ == '__main__':

    # List of tuple of configurations modifications and output names of evaluation
    configurations = [
        ({'cytomine': {'dataset_path': '../datasets/LBTD-NEO04/'}, 'dataset': {'points':True, 'negative_points': True, 'box':False, 'mask_prompt':True}}, 'LBTD-NEO04110'),
        ({'cytomine': {'dataset_path': '../datasets/LBTD-NEO04/'}, 'dataset': {'points':True, 'negative_points': True, 'box':True, 'mask_prompt':True}}, 'LBTD-NEO04111'),
    ]
    output_path = '../output/mask/'
    metrics_path = '../output/mask/metrics/'
    config = load_config('config.toml')

    for configuration, name in configurations:
        config.merge_update(configuration)        
        scores = evaluate_with_config(config)
        dice_scores = scores['dice']
        iou_scores = scores['iou']
        precision_scores = scores['precision']
        recall_scores = scores['recall']
        dice_input_scores = scores['dice_input']
        iou_input_scores = scores['iou_input']
        precision_input_scores = scores['precision_input']
        recall_input_scores = scores['recall_input']
        print(f'{name} dice: {sum(dice_scores)/len(dice_scores)}')
        print(f'{name} iou: {sum(iou_scores)/len(iou_scores)}')
        print(f'{name} precision: {sum(precision_scores)/len(precision_scores)}')
        print(f'{name} recall: {sum(recall_scores)/len(recall_scores)}')

        # Save the results
        with open(f'{output_path}{name}.txt', 'w') as f:
            f.write(f'Mean Dice score: {sum(dice_scores)/len(dice_scores)}\n')
            f.write(f'Mean IoU score: {sum(iou_scores)/len(iou_scores)}\n')
            f.write(f'Mean Precision score: {sum(precision_scores)/len(precision_scores)}\n')
            f.write(f'Mean Recall score: {sum(recall_scores)/len(recall_scores)}\n')
        torch.save(scores, f'{metrics_path}{name}.pt')