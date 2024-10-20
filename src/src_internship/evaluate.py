from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from cv2 import INTER_NEAREST, resize
from dataset_processing.dataset import AugmentedSamDataset, SAMDataset
from dataset_processing.preprocess import collate_fn
from model.model import load_model
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.config import load_config

IMG_RESOLUTION = 1024


def evaluate(dataset_path:str, model_path:str, model_type='vit_b', n_points:int=1, batch_size:int=1, prompt_type:dict={'points':False, 'box':False, 'neg_points':False, 'mask':False}, n_neg_points:int=5, inside_box:bool=False, points_near_center:bool=False, random_box_shift:bool=0, mask_prompt_type:str='truth', box_around_mask:bool=False, input_mask_eval:bool=False, device:str='cuda') -> dict[list]:
    '''Function to evaluate SAM model. 
    dataset_path: str, path to the dataset
    n_points: int, number of points to use in the dataset
    batch_size: int, batch size to use in the evaluation
    prompt_type: dict, dictionary with the prompt type to use in the dataset
    n_neg_points: int, number of negative points to use in the dataset
    inside_box: bool, if True, the negative points are inside the bounding box
    points_near_center: bool, if True, the points are near the center of the annotation
    random_box_shift: bool, if True, the bounding box is randomly shifted
    mask_prompt_type: str, type of mask prompt to use
    box_around_mask: bool, if True, the bounding box is around the input mask
    input_mask_eval: bool, if True, evaluate the input mask too
    model_path: str, path to the model
    device: str, device to use for the evaluation
    Returns: dict, dictionary with the evaluation metrics'''
    dataset = SAMDataset(dataset_path, prompt_type=prompt_type, n_points=n_points, n_neg_points=n_neg_points, verbose=True, to_dict=True, neg_points_inside_box=inside_box, points_near_center=points_near_center, random_box_shift=random_box_shift, mask_prompt_type=mask_prompt_type, box_around_mask=box_around_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model = load_model(model_path, model_type)
    eval_loop(model, dataloader, device, input_mask_eval)
    return scores

def evaluate_with_config(config:dict) -> dict[str,list]:
    '''Function to evaluate a model with a configuration dictionary. Please refers to load_config() function from .utils.config.'''
    prompt_type = {'points':config.dataset.points, 'box':config.dataset.box, 'neg_points':config.dataset.negative_points, 'mask':config.dataset.mask_prompt}
    valid_dataset = AugmentedSamDataset(root=config.evaluate.valid_dataset_path,
                            #prompt_type={'points':config.dataset.points, 'box':config.dataset.box, 'neg_points':config.dataset.negative_points, 'mask':config.dataset.mask_prompt},
                            n_points=config.dataset.n_points,
                            n_neg_points=config.dataset.n_neg_points,
                            verbose=True,
                            to_dict=True,
                            use_img_embeddings=config.training.use_img_embeddings,
                            #neg_points_inside_box=config.dataset.negative_points_inside_box,
                            #points_near_center=config.dataset.points_near_center,
                            random_box_shift=config.dataset.random_box_shift,
                            mask_prompt_type=config.dataset.mask_prompt_type,
                            #box_around_mask=config.dataset.box_around_prompt_mask,
                            load_on_cpu=True
    )
    dataloader = DataLoader(valid_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_fn)
    model = load_model(config.sam.checkpoint_path, config.sam.model_type, img_embeddings_as_input=config.training.use_img_embeddings, return_iou=True)
    scores = test_loop(model, dataloader, config.misc.device, config.evaluate.input_mask_eval, return_mean=False)
    return scores

def eval_loop(model:nn.Module, dataloader:DataLoader, device:str='cuda', input_mask_eval:bool=False, return_mean:bool=True) -> dict:
    '''Function to evaluate a model on a dataloader.
    model: nn.Module, model to evaluate
    dataloader: DataLoader, dataloader to use for the evaluation
    device: str, device to use for the evaluation
    input_mask_eval: bool, if True, evaluate the input mask too
    return_mean: bool, if True, return the mean of the evaluation metrics
    Returns: dict, dictionary with the evaluation metrics'''
    scores = {'BCE':[],'dice':[], 'iou':[], 'precision':[], 'recall':[], 'dice_input':[], 'iou_input':[], 'precision_input':[], 'recall_input':[]}
    model.to(device)
    model.eval()
    model.return_iou = False
    with torch.no_grad():
        for data, mask in tqdm(dataloader):
            pred = model(data, multimask_output=True, binary_mask_output=False)
            loss = nn.BCEWithLogitsLoss()(pred.float(), mask.to(device).float())
            scores['BCE'].append(loss.item())
    if return_mean:
        for key in scores.keys():
            scores['BCE'] = np.mean(scores['BCE'])
    model.return_iou = True
    return scores

def test_loop(model:nn.Module, dataloader:DataLoader, device:str='cuda', input_mask_eval:bool=False, return_mean:bool=True) -> dict:
    scores = {'dice':[], 'iou':[], 'precision':[], 'recall':[], 'dice_input':[], 'iou_input':[], 'precision_input':[], 'recall_input':[]}
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data, mask in tqdm(dataloader):
            pred,_ = model(data, multimask_output=True, binary_mask_output=True)
            best_pred = pred.cpu().numpy()
            best_pred = np.array(best_pred, dtype=np.uint8)
            mask = np.array(mask, dtype=np.uint8)
            for i in range(len(data)):
                y_pred = best_pred[i]
                y_true = mask[i]
                y_true = mask[i].flatten()
                y_pred = best_pred[i].flatten()
                scores['dice'].append(f1_score(y_true, y_pred))
                if input_mask_eval:
                    input_mask = resize(data[i]['mask_inputs'].cpu().numpy()[0][0], (IMG_RESOLUTION, IMG_RESOLUTION), interpolation=INTER_NEAREST)
                    y_input = input_mask.flatten()
                    scores['dice_input'].append(f1_score(y_true, y_input)) 
    if return_mean:
        for key in scores.keys():
            scores[key] = np.mean(scores[key])
    model.return_iou = True
    return scores

if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluate a batch of images using a trained model.')
    parser.add_argument('--config', required=False, type=str, help='Path to the configuration file. Default: ../config.toml', default='../config.toml')
    parser.add_argument('--save_metrics', required=False, type=bool, help='Save the metrics in a .pt file containing a dictionary {"dice", "iou", "precision", "recall"}. Each key is associated to the distribution of the results of evalution over the dataset. Default: False', default=False)
    parser.add_argument('--metrics_path', required=False, type=str, help='Path to save the metrics. Default: metrics.pt', default='')
    args = parser.parse_args()
    config = load_config(args.config)
    scores = evaluate_with_config(config)
    dice_scores = scores['dice']
    iou_scores = scores['iou']
    precision_scores = scores['precision']
    recall_scores = scores['recall']
    print(f'Mean Dice score: {np.mean(dice_scores)}')
    print(f'Mean IoU score: {np.mean(iou_scores)}')
    print(f'Mean Precision score: {np.mean(precision_scores)}')
    print(f'Mean Recall score: {np.mean(recall_scores)}')
    if args.save_metrics:
        torch.save(scores, args.metrics_path)
    

    
    
    



