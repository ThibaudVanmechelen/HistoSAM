import time
import gc
import numpy as np
import torch
import segmentation_refinement as refine
from dataset_processing.dataset import SAMDataset, SamDatasetFromFiles, filter_dataset
from dataset_processing.preprocess import collate_fn
from model.model import load_model
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from model.sam2_model import TrainableSAM2
from model.histo_sam import HistoSAM
from segment_anything import SamAutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from utils.loss import SAM_Loss
from utils.metrics import compute_compactness, compute_perimeter_smoothness_ratio, compute_solidity, compute_size_retention, count_connected_components
from utils.post_processing import post_process_segmentation_mask, post_process_with_crf
from utils.save_scores import save_scores

IMG_RESOLUTION = 1024

def evaluate_standard_SAM_with_config(config : dict, dataset_path : str, checkpoint_path : str, is_sam2 : bool): # config format should be based on prompting evaluation format
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
                        is_sam2_prompt = is_sam2
                    )
    
    dataloader = DataLoader(dataset, batch_size = config.prompting_evaluation.batch_size, shuffle = False, collate_fn = collate_fn)

    if is_sam2 is False:
        model = load_model(checkpoint_path, config.sam.model_type, img_embeddings_as_input = config.prompting_evaluation.use_img_embeddings, return_iou = True)
        scores = test_loop(model, dataloader, config.misc.device, config.prompting_evaluation.input_mask_eval, return_mean = False)
    else:
        model = TrainableSAM2(finetuned_model_name = "prompting_model", cfg = config.sam2.model_type, checkpoint = checkpoint_path, mode = "eval",
                              do_train_prompt_encoder = False, do_train_mask_decoder = False, img_embeddings_as_input = False, device = config.misc.device)
            
        scores = model.test_loop(dataloader, config.prompting_evaluation.input_mask_eval)

    del dataset
    del dataloader
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return scores


def evaluate_histo_SAM_with_config(config : dict, dataset_path : str, checkpoint_paths : list[str], use_dataset : list[bool], model_weight_path : str):
    prompt_type = {
                'points' : config.prompting_evaluation.points, 
                'box' : config.prompting_evaluation.box, 
                'neg_points' : config.prompting_evaluation.negative_points, 
                'mask' : config.prompting_evaluation.mask_prompt
    }

    dataset = SamDatasetFromFiles(
        root = dataset_path,
        transform = None,
        use_img_embeddings = config.prompting_evaluation.use_img_embeddings,
        prompt_type = prompt_type,
        n_points = config.prompting_evaluation.n_points,
        n_neg_points = config.prompting_evaluation.n_neg_points,
        verbose = True,
        to_dict = True,
        is_sam2_prompt = False,
        neg_points_inside_box = config.prompting_evaluation.negative_points_inside_box,
        points_near_center = config.prompting_evaluation.points_near_center,
        random_box_shift = config.prompting_evaluation.random_box_shift,
        mask_prompt_type = config.prompting_evaluation.mask_prompt_type,
        box_around_mask = config.prompting_evaluation.box_around_prompt_mask,
        filter_files = lambda x: filter_dataset(x, use_dataset),
        load_on_cpu = config.prompting_evaluation.load_on_cpu,
        generate_prompt_on_get = config.prompting_evaluation.prompt_on_get,
        is_combined_embedding = config.prompting_evaluation.is_combined_embedding
    )

    dataloader = DataLoader(dataset, batch_size = config.prompting_evaluation.batch_size, shuffle = False, collate_fn = collate_fn)

    model = HistoSAM(model_type = config.sam.model_type,
                    checkpoint_path = checkpoint_paths[0],
                    hist_encoder_type = config.encoder.type,
                    hist_encoder_checkpoint_path = checkpoint_paths[1],
                    not_use_sam_encoder = config.sam.not_use_sam_encoder,
                    embedding_as_input = config.prompting_evaluation.use_img_embeddings,
                    up_sample_with_deconvolution = config.encoder.deconv,
                    freeze_sam_img_encoder = True,
                    freeze_prompt_encoder = True,
                    freeze_mask_decoder = True,
                    return_iou = True,
                    device = config.misc.device                  
    )

    if model_weight_path:
        print("Loading the model weights...")
        state_dict = torch.load(model_weight_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict = False)

        if missing_keys or unexpected_keys:
            print(f"Warning: Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")
        else:
            print("Model loaded successfully!")

    model.eval()
    scores = test_loop(model, dataloader, config.misc.device, config.prompting_evaluation.input_mask_eval, return_mean = False, 
                       is_eval_post_processing = config.prompting_evaluation.is_eval_post_processing, do_post_process = config.prompting_evaluation.do_post_process,
                       post_process_type = config.prompting_evaluation.post_process_type)

    del dataset
    del dataloader
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return scores


def evaluate_SAM_iteratively(configs : list[dict], dataset_path : str, checkpoint_paths : list[str], is_sam2 : bool, output_dirs : list[(str, str)]): # config format should be based on prompting evaluation format
    prompt_type = {
                   'points' : configs[0].prompting_evaluation.points,  # all configs should use the same prompts
                   'box' : configs[0].prompting_evaluation.box, 
                   'neg_points' : configs[0].prompting_evaluation.negative_points, 
                   'mask' : configs[0].prompting_evaluation.mask_prompt
                }

    dataset = SAMDataset(
                        root = dataset_path,
                        prompt_type = prompt_type,
                        n_points = configs[0].prompting_evaluation.n_points,
                        n_neg_points = configs[0].prompting_evaluation.n_neg_points,
                        verbose = True,
                        to_dict = True,
                        use_img_embeddings = configs[0].prompting_evaluation.use_img_embeddings,
                        neg_points_inside_box = configs[0].prompting_evaluation.negative_points_inside_box,
                        points_near_center = configs[0].prompting_evaluation.points_near_center,
                        random_box_shift = configs[0].prompting_evaluation.random_box_shift,
                        mask_prompt_type = configs[0].prompting_evaluation.mask_prompt_type,
                        box_around_mask = configs[0].prompting_evaluation.box_around_prompt_mask,
                        is_sam2_prompt = is_sam2
                    )
    
    dataloader = DataLoader(dataset, batch_size = configs[0].prompting_evaluation.batch_size, shuffle = False, collate_fn = collate_fn)

    for i in range(len(checkpoint_paths)):
        print(f"Starting prompting with {checkpoint_paths[i]}")

        if is_sam2 is False:
            model = load_model(checkpoint_paths[i], configs[i].sam.model_type, img_embeddings_as_input = configs[i].prompting_evaluation.use_img_embeddings, return_iou = True)
            scores = test_loop(model, dataloader, configs[i].misc.device, configs[i].prompting_evaluation.input_mask_eval, return_mean = False)

        else:
            model = TrainableSAM2(finetuned_model_name = "prompting_model", cfg = configs[i].sam2.model_type, checkpoint = checkpoint_paths[i], mode = "eval",
                                do_train_prompt_encoder = False, do_train_mask_decoder = False, img_embeddings_as_input = False, device = configs[i].misc.device)
                
            scores = model.test_loop(dataloader, configs[i].prompting_evaluation.input_mask_eval)
        
        save_scores(scores, output_dirs[i][0], output_dirs[i][1])

        del model
        torch.cuda.empty_cache()
        gc.collect()

    del dataset
    del dataloader
    torch.cuda.empty_cache()
    gc.collect()


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
                        is_sam2_prompt = is_sam2
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


def eval_loop(model : nn.Module, dataloader : DataLoader, device : str = 'cuda', input_mask_eval : bool = False, is_original_loss : bool = False) -> dict:
    """Function to evaluate a model on a dataloader.
    model: nn.Module, model to evaluate
    dataloader: DataLoader, dataloader to use for the evaluation
    device: str, device to use for the evaluation
    input_mask_eval: bool, if True, evaluate the input mask too
    return_mean: bool, if True, return the mean of the evaluation metrics
    Returns: dict, dictionary with the evaluation metrics"""
    scores = {
        'total_loss': [], 'focal_loss': [], 'dice_loss': [], 'iou_loss': [],
        'dice': [], 'iou': [], 'precision': [], 'recall': [], 
        'dice_input': [], 'iou_input': [], 'precision_input': [], 'recall_input': [],
        'prediction_time': []
    }

    model.to(device)
    model.eval()
    model.return_iou = True

    if is_original_loss:
        loss_fn = SAM_Loss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for data, mask in tqdm(dataloader):
            mask = torch.tensor(mask, dtype = torch.float32, device = device)

            start_time = time.time()
            best_pred, pred_iou = model(data, multimask_output = True, binary_mask_output = False)
            end_time = time.time()

            if is_original_loss:
                loss, loss_parts = loss_fn(best_pred.float(), mask.float(), pred_iou.float())

                scores['focal_loss'].append(loss_parts['focal'])
                scores['dice_loss'].append(loss_parts['dice'])
                scores['iou_loss'].append(loss_parts['iou'])
            else:
                loss = loss_fn(best_pred.float(), mask.float())

            scores['total_loss'].append(loss.item())
            scores['prediction_time'].append(end_time - start_time)

            best_pred = torch.where(best_pred > model.mask_threshold, 1, 0).float() # to obtain binary mask for the metrics

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
                input_masks = torch.stack([d['mask_inputs'].to(device).squeeze() for d in data])  # Shape: (N, H, W)
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
                
    return scores


def test_loop(model: torch.nn.Module, dataloader: DataLoader, device: str = 'cuda', input_mask_eval: bool = False, return_mean: bool = True,
              use_automatic_predictor: bool = False, is_sam2: bool = False, is_eval_post_processing : bool = False, do_post_process : bool = False,
              post_process_type : str = 'standard', do_recirculation : bool = False) -> dict:
    scores = {
        'dice': [], 'iou': [], 'precision': [], 'recall': [],
        'dice_input': [], 'iou_input': [], 'precision_input': [], 'recall_input': [],
        'prediction_time': []
    }

    if is_eval_post_processing:
        scores.update({
            'compactness': [], 'solidity': [], 'perimeter_smoothness': [],
            'connected_components': [], 'size_retention': []
        })

    if use_automatic_predictor:
        if not is_sam2:
            predictor = SamAutomaticMaskGenerator(model, points_per_side = 16)
        else:
            predictor = SAM2AutomaticMaskGenerator(model, points_per_side = 16)
    else:
        model.to(device)
        model.eval()
        model.return_iou = False

    if do_post_process and post_process_type == 'cascadepsp':
        refiner = refine.Refiner(device = device)

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

                if do_recirculation:
                    pred = model.predict_with_recirculation(data, multimask_output = True, binary_mask_output = True)
                else:
                    pred = model(data, multimask_output = True, binary_mask_output = True)

                end_time = time.time()
                best_pred = pred

            scores['prediction_time'].append(end_time - start_time)

            if do_post_process:
                best_pred_cpu = best_pred.cpu().numpy().astype(np.uint8) * 255

                if post_process_type == 'standard':
                    processed_masks = np.array([post_process_segmentation_mask(m) for m in best_pred_cpu])

                elif post_process_type == 'densecrf':
                    imgs = [d['image'].permute(1, 2, 0).cpu().numpy() for d in data]
                    processed_masks = []

                    for im in range(len(imgs)):
                        processed_masks.append(post_process_with_crf(imgs[im].astype(np.uint8), best_pred_cpu[im]))

                elif post_process_type == 'cascadepsp':
                    imgs = [d['image'].permute(1, 2, 0).cpu().numpy() for d in data]
                    processed_masks = []

                    for im in range(len(imgs)):
                        processed_masks.append(refiner.refine(imgs[im].astype(np.uint8), best_pred_cpu[im], fast = False, L = 900))

                else:
                    processed_masks = best_pred_cpu 

                for idx in range(len(processed_masks)):
                    scores['size_retention'].append(compute_size_retention(best_pred_cpu[idx], processed_masks[idx]))
                
                processed_masks = np.array(processed_masks)

                if processed_masks.max() > 1:
                    processed_masks = (processed_masks > 127).astype(np.uint8)
                else:
                    processed_masks = (processed_masks > 0).astype(np.uint8)

                best_pred = torch.tensor(processed_masks, device = device)

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
                    input_masks = torch.stack([torch.from_numpy(d['mask_inputs']).to(device).squeeze(0) for d in data])  # Shape: (N, H, W)
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

            if is_eval_post_processing:
                best_pred_cpu = best_pred.cpu().numpy().astype(np.uint8) * 255

                for i in range(best_pred_cpu.shape[0]):
                    processed_mask = best_pred_cpu[i]
                    
                    scores['compactness'].append(compute_compactness(processed_mask))
                    scores['solidity'].append(compute_solidity(processed_mask))
                    scores['perimeter_smoothness'].append(compute_perimeter_smoothness_ratio(processed_mask))
                    scores['connected_components'].append(count_connected_components(processed_mask))

    if not is_sam2 and not use_automatic_predictor:
        model.return_iou = True

    return scores