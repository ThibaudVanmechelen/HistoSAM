from argparse import ArgumentParser
from typing import List

import time
import gc
import numpy as np
import torch
from cv2 import INTER_NEAREST, resize
from dataset_processing.dataset import AugmentedSamDataset, SAMDataset, SamDatasetFromFiles, filter_dataset
from dataset_processing.preprocess import collate_fn
from model.model import load_model
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils.config import load_config
from model.sam2_model import TrainableSAM2
from segment_anything import SamAutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

IMG_RESOLUTION = 1024

def evaluate(dataset_path : str, model_path : str, model_type = 'vit_b', n_points : int = 1, batch_size : int = 1, 
             prompt_type : dict = {'points':False, 'box':False, 'neg_points':False, 'mask':False}, n_neg_points : int = 5, 
             inside_box : bool = False, points_near_center : bool = False, random_box_shift : bool = 0, mask_prompt_type : str = 'truth', 
             box_around_mask : bool = False, input_mask_eval : bool = False, device : str = 'cuda') -> dict[list]:
    """Function to evaluate SAM model. 
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
    Returns: dict, dictionary with the evaluation metrics"""
    dataset = SAMDataset(dataset_path, prompt_type=prompt_type, n_points=n_points, n_neg_points=n_neg_points, verbose=True, to_dict=True, neg_points_inside_box=inside_box, points_near_center=points_near_center, random_box_shift=random_box_shift, mask_prompt_type=mask_prompt_type, box_around_mask=box_around_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = load_model(model_path, model_type)
    scores = eval_loop(model, dataloader, device, input_mask_eval)

    return scores


# def evaluate_with_config(config : dict) -> dict[str,list]:
#     """Function to evaluate a model with a configuration dictionary. Please refers to load_config() function from .utils.config."""
#     prompt_type = {'points':config.dataset.points, 'box':config.dataset.box, 'neg_points':config.dataset.negative_points, 'mask':config.dataset.mask_prompt}
#     valid_dataset = AugmentedSamDataset(root=config.evaluate.valid_dataset_path,
#                             #prompt_type={'points':config.dataset.points, 'box':config.dataset.box, 'neg_points':config.dataset.negative_points, 'mask':config.dataset.mask_prompt},
#                             n_points=config.dataset.n_points,
#                             n_neg_points=config.dataset.n_neg_points,
#                             verbose=True,
#                             to_dict=True,
#                             use_img_embeddings=config.training.use_img_embeddings,
#                             #neg_points_inside_box=config.dataset.negative_points_inside_box,
#                             #points_near_center=config.dataset.points_near_center,
#                             random_box_shift=config.dataset.random_box_shift,
#                             mask_prompt_type=config.dataset.mask_prompt_type,
#                             #box_around_mask=config.dataset.box_around_prompt_mask,
#                             load_on_cpu=True
#     )
#     dataloader = DataLoader(valid_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_fn)

#     model = load_model(config.sam.checkpoint_path, config.sam.model_type, img_embeddings_as_input=config.training.use_img_embeddings, return_iou=True)
#     scores = test_loop(model, dataloader, config.misc.device, config.evaluate.input_mask_eval, return_mean=False)

#     return scores

def evaluate_standard_SAM_with_config(config : dict, dataset_path : str, checkpoint_path : str, is_sam2 : bool): #config format should be based on prompting evaluation format
    prompt_type = {
                   'points' : config.prompting_evaluation.points, 
                   'box' : config.prompting_evaluation.box, 
                   'neg_points' : config.prompting_evaluation.negative_points, 
                   'mask' : config.prompting_evaluation.mask_prompt
                }
    
    dataset = SAMDataset(
                        root = dataset_path,
                        prompt_type = prompt_type,
                        n_points = config.prompting_evaluation.n_points,
                        n_neg_points = config.prompting_evaluation.n_neg_points,
                        verbose = True,
                        to_dict = True,
                        use_img_embeddings = config.prompting_evaluation.use_img_embeddings,
                        neg_points_inside_box = config.prompting_evaluation.negative_points_inside_box,
                        points_near_center = config.prompting_evaluation.points_near_center,
                        random_box_shift = config.prompting_evaluation.random_box_shift,
                        mask_prompt_type = config.prompting_evaluation.mask_prompt_type,
                        box_around_mask = config.prompting_evaluation.box_around_prompt_mask,
                    )
    
    dataloader = DataLoader(dataset, batch_size = config.prompting_evaluation.batch_size, shuffle = False, collate_fn = collate_fn)

    if is_sam2 is False:
        model = load_model(checkpoint_path, config.sam.model_type, img_embeddings_as_input = config.prompting_evaluation.use_img_embeddings, return_iou = True)
        scores = test_loop(model, dataloader, config.misc.device, config.prompting_evaluation.input_mask_eval, return_mean = False)
    else:
        model = TrainableSAM2(finetuned_model_name = "prompting_model", cfg = config.sam2.model_type, checkpoint = checkpoint_path, mode = "eval",
                              do_train_prompt_encoder = False, do_train_mask_decoder = False, img_embeddings_as_input = False, device = config.misc.device)
            
        scores = model.eval_loop(dataloader, config.prompting_evaluation.input_mask_eval, return_mean = False)


    del dataset
    del dataloader
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return scores


def evaluate_without_prompts(dataset_path : str, checkpoint_path : str, is_sam2 : bool, model_type : str, device : str):
    prompt_type = {
                   'points' : False, 
                   'box' : False, 
                   'neg_points' : False, 
                   'mask' : False
                }
    
    dataset = SAMDataset(
                        root = dataset_path,
                        prompt_type = prompt_type,
                        n_points = 0,
                        n_neg_points = 0,
                        verbose = True,
                        to_dict = True,
                        use_img_embeddings = False,
                        neg_points_inside_box = False,
                        points_near_center = -1,
                        random_box_shift = 0,
                        mask_prompt_type = 'scribble',
                        box_around_mask = False,
                    )
    
    dataloader = DataLoader(dataset, batch_size = 4, shuffle = False, collate_fn = collate_fn)

    if is_sam2 is False:
        model = load_model(checkpoint_path, model_type, img_embeddings_as_input = False, return_iou = True).to(device)
    else:
        model = build_sam2(model_type, checkpoint_path, device = device, apply_postprocessing = True)

    scores = test_loop(model, dataloader, device, False, return_mean = False, use_automatic_predictor = True, is_sam2 = is_sam2)

    del dataset
    del dataloader
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return scores



def evaluate_with_config(config : dict, use_dataset : List[bool] = [True, True, True]): #TODO check if fromFiles is the right class for testing vs augmented dataset ?
    """Function to evaluate a model with a configuration dictionary. Please refers to load_config() function from .utils.config."""
    prompt_type = {'points':config.dataset.points, 'box':config.dataset.box, 'neg_points':config.dataset.negative_points, 'mask':config.dataset.mask_prompt}
    test_dataset = SamDatasetFromFiles(root=config.evaluate.valid_dataset_path,
                            #prompt_type={'points':config.dataset.points, 'box':config.dataset.box, 'neg_points':config.dataset.negative_points, 'mask':config.dataset.mask_prompt},
                            n_points=config.dataset.n_points,
                            n_neg_points=config.dataset.n_neg_points,
                            verbose=True,
                            to_dict=True,
                            use_img_embeddings=config.training.use_img_embeddings,
                            random_box_shift=config.dataset.random_box_shift,
                            mask_prompt_type=config.dataset.mask_prompt_type,
                            #box_around_mask=config.dataset.box_around_prompt_mask,
                            load_on_cpu=True,
                            filter_files=lambda x: filter_dataset(x, use_dataset)
    )
    dataloader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_fn)

    model = load_model(config.sam.checkpoint_path, config.sam.model_type, img_embeddings_as_input=config.training.use_img_embeddings, return_iou=True)
    scores = test_loop(model, dataloader, config.misc.device, config.evaluate.input_mask_eval, return_mean=False)

    return scores


# def eval_loop(model : nn.Module, dataloader : DataLoader, device : str='cuda', input_mask_eval : bool=False, return_mean : bool=True) -> dict:
#     """Function to evaluate a model on a dataloader.
#     model: nn.Module, model to evaluate
#     dataloader: DataLoader, dataloader to use for the evaluation
#     device: str, device to use for the evaluation
#     input_mask_eval: bool, if True, evaluate the input mask too
#     return_mean: bool, if True, return the mean of the evaluation metrics
#     Returns: dict, dictionary with the evaluation metrics"""
#     scores = {
#               'BCE':[], 'prediction_time':[],
#               'dice':[], 'iou':[], 'precision':[], 'recall':[], 
#               'dice_input':[], 'iou_input':[], 'precision_input':[], 'recall_input':[]
#              }

#     model.to(device)
#     model.eval()
#     model.return_iou = False

#     with torch.no_grad():
#         for data, mask in tqdm(dataloader):
#             start_time = time.time()
#             pred = model(data, multimask_output=True, binary_mask_output=False)
#             end_time = time.time()

#             loss = nn.BCEWithLogitsLoss()(pred.float(), mask.to(device).float())

#             scores['BCE'].append(loss.item())
#             scores['prediction_time'].append(end_time - start_time)

#             best_pred = pred.cpu().numpy()
#             best_pred = np.array(best_pred, dtype = np.uint8)

#             mask = np.array(mask, dtype = np.uint8)
#             for i in range(len(data)): # going through each item in the batch
#                 y_pred = best_pred[i].flatten()
#                 y_true = mask[i].flatten()
                
#                 scores['dice'].append(f1_score(y_true, y_pred))
#                 scores['iou'].append(jaccard_score(y_true, y_pred))
#                 scores['precision'].append(precision_score(y_true, y_pred, zero_division = 1))
#                 scores['recall'].append(recall_score(y_true, y_pred, zero_division = 1))

#                 if input_mask_eval:
#                     input_mask = resize(data[i]['mask_inputs'].cpu().numpy()[0][0], (IMG_RESOLUTION, IMG_RESOLUTION), interpolation = INTER_NEAREST)
#                     y_input = input_mask.flatten()

#                     scores['dice_input'].append(f1_score(y_true, y_input)) 
#                     scores['iou_input'].append(jaccard_score(y_true, y_input))
#                     scores['precision_input'].append(precision_score(y_true, y_input, zero_division = 1))
#                     scores['recall_input'].append(recall_score(y_true, y_input, zero_division = 1))

#     if return_mean:
#         for key in scores.keys():
#             scores[key] = np.mean(scores[key])

#     model.return_iou = True

#     return scores

def eval_loop(model : nn.Module, dataloader : DataLoader, device : str='cuda', input_mask_eval : bool=False, return_mean : bool=True) -> dict:
    """Function to evaluate a model on a dataloader.
    model: nn.Module, model to evaluate
    dataloader: DataLoader, dataloader to use for the evaluation
    device: str, device to use for the evaluation
    input_mask_eval: bool, if True, evaluate the input mask too
    return_mean: bool, if True, return the mean of the evaluation metrics
    Returns: dict, dictionary with the evaluation metrics"""
    scores = {
              'BCE':[], 'prediction_time':[],
              'dice':[], 'iou':[], 'precision':[], 'recall':[], 
              'dice_input':[], 'iou_input':[], 'precision_input':[], 'recall_input':[]
             }

    model.to(device)
    model.eval()
    model.return_iou = False

    with torch.no_grad():
        for data, mask in tqdm(dataloader):
            start_time = time.time()
            pred = model(data, multimask_output = True, binary_mask_output = False)
            end_time = time.time()

            loss = nn.BCEWithLogitsLoss()(pred.float(), mask.to(device).float())

            scores['BCE'].append(loss.item())
            scores['prediction_time'].append(end_time - start_time)

            best_pred = pred.cpu().numpy()
            best_pred = np.array(best_pred, dtype = np.uint8)

            mask = np.array(mask, dtype = np.uint8)
            for i in range(len(data)): # going through each item in the batch
                y_pred = best_pred[i].flatten()
                y_true = mask[i].flatten()
                
                scores['dice'].append(f1_score(y_true, y_pred))
                scores['iou'].append(jaccard_score(y_true, y_pred))
                scores['precision'].append(precision_score(y_true, y_pred, zero_division = 1))
                scores['recall'].append(recall_score(y_true, y_pred, zero_division = 1))

                if input_mask_eval:
                    input_mask = resize(data[i]['mask_inputs'].cpu().numpy()[0][0], (IMG_RESOLUTION, IMG_RESOLUTION), interpolation = INTER_NEAREST)
                    y_input = input_mask.flatten()

                    scores['dice_input'].append(f1_score(y_true, y_input)) 
                    scores['iou_input'].append(jaccard_score(y_true, y_input))
                    scores['precision_input'].append(precision_score(y_true, y_input, zero_division = 1))
                    scores['recall_input'].append(recall_score(y_true, y_input, zero_division = 1))

    if return_mean:
        for key in scores.keys():
            scores[key] = np.mean(scores[key])

    model.return_iou = True

    return scores

# def test_loop(model : nn.Module, dataloader : DataLoader, device : str = 'cuda', input_mask_eval : bool = False, return_mean : bool = True, 
#               use_automatic_predictor : bool = False, is_sam2 : bool = False) -> dict:
#     scores = {
#               'dice':[], 'iou':[], 'precision':[], 'recall':[], 
#               'dice_input':[], 'iou_input':[], 'precision_input':[], 'recall_input':[],
#               'prediction_time':[]
#              }

#     if use_automatic_predictor:
#         if is_sam2 is False:
#             predictor = SamAutomaticMaskGenerator(model)
#         else:
#             predictor = SAM2AutomaticMaskGenerator(model)
#     else:
#         model.to(device)
#         model.eval()
#         model.return_iou = False

#     with torch.no_grad():
#         for data, mask in tqdm(dataloader):
#             if use_automatic_predictor:
#                 best_pred = []
#                 unprocessed_imgs = []

#                 for u in range(len(data)):
#                     unprocessed_imgs.append(data[u]['image'].permute(1, 2, 0).cpu().numpy())

#                 start_time = time.time()
#                 for u in range(len(data)):
#                     pred_masks = predictor.generate(unprocessed_imgs[u])
#                     best_mask = max(pred_masks, key = lambda x: x['predicted_iou'])['segmentation']
#                     best_pred.append(best_mask)

#                 end_time = time.time()

#             else:
#                 start_time = time.time()
#                 pred = model(data, multimask_output = True, binary_mask_output = True)
#                 end_time = time.time()

#                 best_pred = pred.cpu().numpy()

#             scores['prediction_time'].append(end_time - start_time)
#             best_pred = np.array(best_pred, dtype = np.uint8)

#             mask = np.array(mask, dtype = np.uint8)
#             for i in range(len(data)):
#                 y_pred = np.where(best_pred[i].flatten() > 0, 1, 0)
#                 y_true = np.where(mask[i].flatten() > 0, 1, 0)

#                 scores['dice'].append(f1_score(y_true, y_pred))
#                 scores['iou'].append(jaccard_score(y_true, y_pred))
#                 scores['precision'].append(precision_score(y_true, y_pred, zero_division = 1))
#                 scores['recall'].append(recall_score(y_true, y_pred, zero_division = 1))

#                 if input_mask_eval:
#                     input_mask = resize(data[i]['mask_inputs'].cpu().numpy()[0][0], (IMG_RESOLUTION, IMG_RESOLUTION), interpolation = INTER_NEAREST)
#                     y_input = np.where(input_mask.flatten() > 0, 1, 0)

#                     scores['dice_input'].append(f1_score(y_true, y_input)) 
#                     scores['iou_input'].append(jaccard_score(y_true, y_input))
#                     scores['precision_input'].append(precision_score(y_true, y_input, zero_division = 1))
#                     scores['recall_input'].append(recall_score(y_true, y_input, zero_division = 1))

#     if return_mean:
#         for key in scores.keys():
#             scores[key] = np.mean(scores[key])

#     model.return_iou = True

#     return scores


def test_loop(model: torch.nn.Module, dataloader: DataLoader, device: str = 'cuda', input_mask_eval: bool = False, return_mean: bool = True,
              use_automatic_predictor: bool = False, is_sam2: bool = False) -> dict:
    scores = {
        'dice': [], 'iou': [], 'precision': [], 'recall': [],
        'dice_input': [], 'iou_input': [], 'precision_input': [], 'recall_input': [],
        'prediction_time': []
    }

    if use_automatic_predictor:
        if not is_sam2:
            predictor = SamAutomaticMaskGenerator(model, points_per_side = 16)
        else:
            predictor = SAM2AutomaticMaskGenerator(model)
    else:
        model.to(device)
        model.eval()
        model.return_iou = False

    with torch.no_grad():
        for data, mask in tqdm(dataloader):
            mask = torch.tensor(mask, dtype = torch.float32, device = device)

            if use_automatic_predictor:
                best_pred = []
                unprocessed_imgs = [d['image'].permute(1, 2, 0).cpu().numpy() for d in data]

                start_time = time.time()
                for u in range(len(data)):
                    pred_masks = predictor.generate(unprocessed_imgs[u])

                    if pred_masks:
                        best_mask = max(pred_masks, key = lambda x: x['predicted_iou'])['segmentation']
                    else:
                        best_mask = np.zeros((IMG_RESOLUTION, IMG_RESOLUTION), dtype = np.uint8)

                    best_pred.append(best_mask)

                end_time = time.time()
                best_pred = torch.tensor(best_pred, dtype = torch.float32, device = device)
            else:
                start_time = time.time()
                pred = model(data, multimask_output = True, binary_mask_output = True)
                end_time = time.time()
                best_pred = pred

            scores['prediction_time'].append(end_time - start_time)
            y_true_flat = mask.view(mask.size(0), -1)
            y_pred_flat = best_pred.view(best_pred.size(0), -1)

            y_true_sum = y_true_flat.sum(dim = 1)
            y_pred_sum = y_pred_flat.sum(dim = 1)

            intersection = (y_true_flat * y_pred_flat).sum(dim = 1)
            union = y_true_sum + y_pred_sum - intersection

            dice_score = (2 * intersection) / (y_true_sum + y_pred_sum)
            iou_score = intersection / union
            precision = intersection / y_pred_sum
            recall = intersection / y_true_sum

            dice_score = torch.nan_to_num(dice_score, nan = 0.0)
            iou_score = torch.nan_to_num(iou_score, nan = 0.0)
            precision = torch.nan_to_num(precision, nan = 0.0)
            recall = torch.nan_to_num(recall, nan = 0.0)

            scores['dice'].extend(dice_score.cpu().numpy())
            scores['iou'].extend(iou_score.cpu().numpy())
            scores['precision'].extend(precision.cpu().numpy())
            scores['recall'].extend(recall.cpu().numpy())

            if input_mask_eval:
                if not is_sam2:
                    input_masks = torch.stack([d['mask_inputs'].to(device).squeeze() for d in data])  # Shape: (N, H, W)
                    input_masks = input_masks.unsqueeze(1)  # Shape: (N, 1, H, W)

                else:
                    input_masks = torch.stack([torch.from_numpy(d['mask_inputs']).to(device).squeeze(0)for d in data])  # Shape: (N, H, W)
                    input_masks = input_masks.unsqueeze(1)  # Shape: (N, 1, H, W)

                resized_masks = F.interpolate(input_masks, size = (IMG_RESOLUTION, IMG_RESOLUTION), mode = 'nearest-exact')
                resized_masks_flat = resized_masks.view(resized_masks.size(0), -1)

                y_input_sum = resized_masks_flat.sum(dim = 1)

                intersection_input = (y_true_flat * resized_masks_flat).sum(dim = 1)
                union_input = y_true_sum + y_input_sum - intersection_input

                dice_input = (2 * intersection_input) / (y_true_sum + y_input_sum)
                iou_input = intersection_input / union_input
                precision_input = intersection_input / y_input_sum
                recall_input = intersection_input / y_true_sum

                dice_input = torch.nan_to_num(dice_input, nan = 0.0)
                iou_input = torch.nan_to_num(iou_input, nan = 0.0)
                precision_input = torch.nan_to_num(precision_input, nan = 0.0)
                recall_input = torch.nan_to_num(recall_input, nan = 0.0)

                scores['dice_input'].extend(dice_input.cpu().numpy())
                scores['iou_input'].extend(iou_input.cpu().numpy())
                scores['precision_input'].extend(precision_input.cpu().numpy())
                scores['recall_input'].extend(recall_input.cpu().numpy())

    if not is_sam2 and not use_automatic_predictor:
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
    prediction_times = scores['prediction_time']

    print(f'Mean Dice score: {np.mean(dice_scores)}')
    print(f'Mean IoU score: {np.mean(iou_scores)}')
    print(f'Mean Precision score: {np.mean(precision_scores)}')
    print(f'Mean Recall score: {np.mean(recall_scores)}')
    print(f'Mean Prediction time: {np.mean(prediction_times)}')

    if args.save_metrics:
        torch.save(scores, args.metrics_path)