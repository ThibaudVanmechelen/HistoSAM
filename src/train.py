"""This script allows to train a SAM model on a dataset. The dataset should be in the format of the SAM dataset."""
import os
import torch
import wandb
from dataset_processing.dataset import (
    AugmentedSamDataset,
    SamDatasetFromFiles,
    filter_dataset,
)
from dataset_processing.preprocess import collate_fn
from .evaluate import eval_loop
from model.model import load_model
from segment_anything.modeling.sam import Sam
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.loss import SAM_Loss

def train_with_config(config: dict, checkpoint_path : str, training_dataset_path : str, validation_dataset_path : str, use_original_sam_loss : bool) -> dict:
    model = load_model(checkpoint_path, config.sam.model_type, img_embeddings_as_input = config.training.use_img_embeddings, return_iou = True).to(config.misc.device)
    print(f"Model parameters: {model.get_nb_parameters(img_encoder=True) / 1e6:.2f}M")

    prompt_type = {
        'points' : config.training.points, 
        'box' : config.training.box, 
        'neg_points' : config.training.negative_points, 
        'mask' : config.training.mask_prompt
    }

    train_dataset = SamDatasetFromFiles(
        root = training_dataset_path,
        transform = None,
        use_img_embeddings = config.training.use_img_embeddings,
        prompt_type = prompt_type,
        n_points = config.training.n_points,
        n_neg_points = config.training.n_neg_points,
        verbose = True,
        to_dict = True,
        is_sam2_prompt = False,
        neg_points_inside_box = config.training.negative_points_inside_box,
        points_near_center = config.training.points_near_center,
        random_box_shift = config.training.random_box_shift,
        mask_prompt_type = config.training.mask_prompt_type,
        box_around_mask = config.training.box_around_prompt_mask,
        filter_files = None,
        load_on_cpu = True
    )

    trainloader = DataLoader(train_dataset, batch_size = config.training.batch_size, shuffle = True, collate_fn = collate_fn)

    if validation_dataset_path is not None:
        valid_dataset = SamDatasetFromFiles(
            root = validation_dataset_path,
            transform = None,
            use_img_embeddings = config.validation.use_img_embeddings,
            prompt_type = prompt_type,
            n_points = config.training.n_points,
            n_neg_points = config.training.n_neg_points,
            verbose = True,
            to_dict = True,
            is_sam2_prompt = False,
            neg_points_inside_box = config.training.negative_points_inside_box,
            points_near_center = config.training.points_near_center,
            random_box_shift = config.training.random_box_shift,
            mask_prompt_type = config.training.mask_prompt_type,
            box_around_mask = config.training.box_around_prompt_mask,
            filter_files = None,
            load_on_cpu = True
        )

        validloader = DataLoader(valid_dataset, batch_size = config.training.batch_size, shuffle = False, collate_fn = collate_fn)
    else:
        validloader = None

    if config.misc.wandb:
        wandb.init(project = config.misc.project_name, config = config)

    if use_original_sam_loss:
        loss_fn = SAM_Loss()
    else:
        loss_fn = BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(list(model.mask_decoder.parameters()) + list(model.prompt_encoder.parameters()), lr = config.training.lr, betas = (0.9, 0.999), weight_decay = 1e-4) # original SAM parameters
    # optimizer = torch.optim.Adam(list(model.mask_decoder.parameters()) + list(model.prompt_encoder.parameters()),lr=8e-4, betas=(0.9, 0.999))

    for param in model.image_encoder.parameters():
        param.requires_grad = False

    for param in model.mask_decoder.parameters():
        param.requires_grad = True

    for param in model.prompt_encoder.parameters():
        param.requires_grad = True

    if config.training.train_from_last_checkpoint:
        checkpoint = torch.load(config.training.last_checkpoint_path)

        model.load_state_dict(torch.load(config.training.last_model_path))
        optimizer.load_state_dict(checkpoint['optimizer'])

        return train_loop(model, trainloader, optimizer, config.training.epochs, loss_fn, validloader, config.training.model_save_dir, 
                                          config.misc.device, use_wandb = config.misc.wandb, last_epoch = checkpoint['epoch'], eval_frequency = config.validation.frequency, is_original_loss = use_original_sam_loss)
    
    return train_loop(model, trainloader, optimizer, config.training.epochs, loss_fn, validloader, config.training.model_save_dir, 
                      config.misc.device, use_wandb = config.misc.wandb, last_epoch = -1, eval_frequency = config.validation.frequency, is_original_loss = use_original_sam_loss)

def data_to_gpu(data : list[dict], device : str = 'cuda') -> list[dict]:
    """Move data to a device."""
    for value in data:
        for key in value:
            if type(value[key]) == torch.Tensor :
                value[key] = value[key].to(device)

    return data

def train_loop(model : Sam, trainloader : DataLoader, optimizer : Optimizer, epochs : int, loss_fn : callable, 
               evalloader : DataLoader = None, model_save_dir : str = None, device : str = 'cpu', verbose : bool = True, 
               use_wandb : bool = False, last_epoch : int = -1, eval_frequency : int = 1, is_original_loss : bool = False) -> dict:
    """Function to train a model on a dataloader.
    model: nn.Module, model to train
    trainloader: DataLoader, dataloader to use for the training
    optimizer: Adam, optimizer to use for the training
    epochs: int, number of epochs to train the model
    evalloader: DataLoader, If provided, evaluate the model at each epochs on it. Default: None
    model_save_dir: str, If provided, save the model at each epochs. Also save the best model (evaluation loss) if evalloader is provided. Default: None
    device: str, device to use for the training
    Returns: dict, dictionary with the training metrics"""
    best_loss = float('inf')
    model.return_iou = True

    scores = {
        'training_total_loss': [], 'training_focal_loss': [], 'training_dice_loss': [], 'training_iou_loss': [],
        'validation_total_loss': [], 'validation_focal_loss': [], 'validation_dice_loss': [], 'validation_iou_loss': [],
        'validation_dice': [], 'validation_iou': [], 'validation_precision': [], 'validation_recall': [], 'validation_prediction_time': []
    }
    
    start_epoch = last_epoch + 1 if last_epoch >= 0 else 0
    for epoch in range(start_epoch, epochs):
        total_losses = []
        focal_losses = []
        dice_losses = []
        iou_losses = []

        model.train()
        for data, mask in tqdm(trainloader, disable = not verbose, desc = f'Epoch {epoch + 1}/{epochs} - Training'):
            data = data_to_gpu(data, device)
            mask = mask.to(device)
    
            optimizer.zero_grad()
            pred_model, pred_iou = model(data, multimask_output = True, binary_mask_output = False)

            if is_original_loss:
                loss, loss_parts = loss_fn(pred_model.float(), mask.float(), pred_iou.float())

                focal_losses.append(loss_parts['focal'])
                dice_losses.append(loss_parts['dice'])
                iou_losses.append(loss_parts['iou'])
            else:
                loss = loss_fn(pred_model.float(), mask.float())

            loss.backward()

            optimizer.step()
            total_losses.append(loss.item())
        
        mean_total_loss = sum(total_losses) / len(total_losses)
        scores['training_total_loss'].append(mean_total_loss)

        if is_original_loss:
            mean_focal_loss = sum(focal_losses) / len(focal_losses)
            mean_dice_loss = sum(dice_losses) / len(dice_losses)
            mean_iou_loss = sum(iou_losses) / len(iou_losses)

            scores['training_focal_loss'].append(mean_focal_loss)
            scores['training_dice_loss'].append(mean_dice_loss)
            scores['training_iou_loss'].append(mean_iou_loss)

        if verbose:
            if is_original_loss:
                print(f'Training - Mean Total Loss: {mean_total_loss}, Focal Loss: {mean_focal_loss}, Dice Loss: {mean_dice_loss}, IoU: {mean_iou_loss}')
            else:
                print(f'Training - Mean Total Loss: {mean_total_loss}')

        if use_wandb:
            if is_original_loss:
                wandb.log({'total_loss': mean_total_loss, 'focal_loss': mean_focal_loss, 'dice_loss': mean_dice_loss, 'iou_loss': mean_iou_loss})
            else:
                wandb.log({'total_loss': mean_total_loss})

        if model_save_dir is not None:
            model_save_path = os.path.join(model_save_dir, 'last_model.pt')
            checkpoint_save_path = os.path.join(model_save_dir, 'last_checkpoint.pt')

            torch.save(model.state_dict(), model_save_path)
            torch.save({'epoch': epoch, 'optimizer': optimizer.state_dict()}, checkpoint_save_path)

        if evalloader is not None and epoch % eval_frequency == 0:
            scores_eval = eval_loop(model, evalloader, device, is_original_loss = is_original_loss)
            scores_eval = { key: sum(value) / len(value) if value else 0 for key, value in scores_eval.items() }

            if is_original_loss:
                print(f'''Evaluation - Total Loss: {scores_eval["total_loss"]}, 
                                        Focal Loss: {scores_eval["focal_loss"]},
                                        Dice Loss: {scores_eval["dice_loss"]},
                                        IoU Loss: {scores_eval["iou_loss"]},
                                        Dice: {scores_eval["dice"]}, 
                                        IoU: {scores_eval["iou"]}, 
                                        Precision: {scores_eval["precision"]}, 
                                        Recall: {scores_eval["recall"]}, 
                                        Time: {scores_eval["prediction_time"]}''')

                if use_wandb:
                    wandb.log({"eval_total_loss": scores_eval["total_loss"],
                               "eval_focal_loss": scores_eval["focal_loss"],
                               "eval_dice_loss": scores_eval["dice_loss"],
                               "eval_iou_loss": scores_eval["iou_loss"],
                               "eval_dice": scores_eval["dice"],
                               "eval_iou": scores_eval["iou"],
                               "eval_precision": scores_eval["precision"], 
                               "eval_recall": scores_eval["recall"], 
                               "eval_time": scores_eval["prediction_time"]})
                    
                scores['validation_total_loss'].append(scores_eval["total_loss"])
                scores['validation_focal_loss'].append(scores_eval["focal_loss"])
                scores['validation_dice_loss'].append(scores_eval["dice_loss"])
                scores['validation_iou_loss'].append(scores_eval["iou_loss"])
                scores['validation_dice'].append(scores_eval["dice"])
                scores['validation_iou'].append(scores_eval["iou"])
                scores['validation_precision'].append(scores_eval["precision"])
                scores['validation_recall'].append(scores_eval["recall"])
                scores['validation_prediction_time'].append(scores_eval["prediction_time"])

            else:
                print(f'''Evaluation - Total Loss: {scores_eval["total_loss"]}, 
                                        Dice: {scores_eval["dice"]}, 
                                        IoU: {scores_eval["iou"]}, 
                                        Precision: {scores_eval["precision"]}, 
                                        Recall: {scores_eval["recall"]}, 
                                        Time: {scores_eval["prediction_time"]}''')

                if use_wandb:
                    wandb.log({"eval_total_loss": scores_eval["total_loss"],
                               "eval_dice": scores_eval["dice"],
                               "eval_iou": scores_eval["iou"],
                               "eval_precision": scores_eval["precision"], 
                               "eval_recall": scores_eval["recall"], 
                               "eval_time": scores_eval["prediction_time"]})
                    
                scores['validation_total_loss'].append(scores_eval["total_loss"])
                scores['validation_dice'].append(scores_eval["dice"])
                scores['validation_iou'].append(scores_eval["iou"])
                scores['validation_precision'].append(scores_eval["precision"])
                scores['validation_recall'].append(scores_eval["recall"])
                scores['validation_prediction_time'].append(scores_eval["prediction_time"])

            if best_loss > scores_eval["total_loss"]:
                best_loss = scores_eval["total_loss"]
                best_model_save_path = os.path.join(model_save_dir, 'best_model.pt')
                
                torch.save(model.state_dict(), best_model_save_path)

    return scores