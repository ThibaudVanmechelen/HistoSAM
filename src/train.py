"""This script allows to train a SAM model on a dataset. The dataset should be in the format of the SAM dataset."""
from argparse import ArgumentParser
from typing import List

import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import wandb
from dataset_processing.dataset import (
    AugmentedSamDataset,
    SamDatasetFromFiles,
    filter_dataset,
)
from dataset_processing.preprocess import collate_fn
from evaluate import eval_loop
from model.model import load_model
from segment_anything.modeling.sam import Sam
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils.config import load_config
from utils.loss import SAM_Loss


# def train_with_config(config : dict, use_dataset : List[bool] = [True, True, False]):
#     """Train a model with a configuration dictionary. Please refers to load_config() function from .utils.config."""
#     model = load_model(config.sam.checkpoint_path, config.sam.model_type, img_embeddings_as_input=config.training.use_img_embeddings, return_iou=True).to('cuda')
#     print(model.get_nb_parameters(img_encoder = True))

#     model = nn.DataParallel(model) # useful when multiple GPUs are available

#     train_dataset = SamDatasetFromFiles(root=config.cytomine.dataset_path,
#                             #prompt_type={'points':config.dataset.points, 'box':config.dataset.box, 'neg_points':config.dataset.negative_points, 'mask':config.dataset.mask_prompt},
#                             n_points=config.dataset.n_points,
#                             n_neg_points=config.dataset.n_neg_points,
#                             verbose=True,
#                             to_dict=True,
#                             use_img_embeddings=config.training.use_img_embeddings,
#                             random_box_shift=config.dataset.random_box_shift,
#                             mask_prompt_type=config.dataset.mask_prompt_type,
#                             #box_around_mask=config.dataset.box_around_prompt_mask,
#                             load_on_cpu=True,
#                             filter_files=lambda x: filter_dataset(x, use_dataset)
#     )

#     valid_dataset = SamDatasetFromFiles(root=config.evaluate.valid_dataset_path,
#                             #prompt_type={'points':config.dataset.points, 'box':config.dataset.box, 'neg_points':config.dataset.negative_points, 'mask':config.dataset.mask_prompt},
#                             n_points=config.dataset.n_points,
#                             n_neg_points=config.dataset.n_neg_points,
#                             verbose=True,
#                             to_dict=True,
#                             use_img_embeddings=config.training.use_img_embeddings,
#                             random_box_shift=config.dataset.random_box_shift,
#                             mask_prompt_type=config.dataset.mask_prompt_type,
#                             load_on_cpu=True,
#                             filter_files=lambda x: filter_dataset(x, use_dataset)
#     )

#     if config.misc.wandb:
#         wandb.init(project='samsam',
#                     config=config)

#     trainloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, collate_fn=collate_fn)
#     validloader = DataLoader(valid_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_fn)

#     optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
#     loss_fn = BCEWithLogitsLoss()

#     if config.training.train_from_last_checkpoint:
#         checkpoint = torch.load(config.training.model_save_dir+'/last_checkpoint.pt')

#         model.load_state_dict(torch.load(config.training.model_save_dir+'/last_model.pt'))
#         optimizer.load_state_dict(checkpoint['optimizer'])

#         return train_from_last_checkpoint(model, trainloader, optimizer, config.training.epochs, loss_fn, validloader, config.training.model_save_dir, config.misc.device, use_wandb=config.misc.wandb, last_epoch=checkpoint['epoch'])
    
#     if config.training.eval_every_epoch:
#         return train_loop(model, trainloader, optimizer, config.training.epochs, loss_fn, validloader, config.training.model_save_dir, config.misc.device, use_wandb=config.misc.wandb)
#     else:
#         return train_loop(model, trainloader, optimizer, config.training.epochs, loss_fn, None, config.training.model_save_dir, config.misc.device, use_wandb=config.misc.wandb)

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
        load_on_cpu = True,
        prompt_path = config.training.prompt_path,
        img_embeddings_path = config.training.img_embeddings_path
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
            load_on_cpu = True,
            prompt_path = config.validation.prompt_path,
            img_embeddings_path = config.validation.img_embeddings_path
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

        return train_from_last_checkpoint(model, trainloader, optimizer, config.training.epochs, loss_fn, validloader, config.training.model_save_dir, 
                                          config.misc.device, use_wandb = config.misc.wandb, last_epoch = checkpoint['epoch'], is_original_loss = use_original_sam_loss)
    
    return train_loop(model, trainloader, optimizer, config.training.epochs, loss_fn, validloader, config.training.model_save_dir, 
                      config.misc.device, use_wandb = config.misc.wandb, is_original_loss = use_original_sam_loss)

def data_to_gpu(data : list[dict], device : str = 'cuda') -> list[dict]:
    """Move data to a device."""
    for value in data:
        for key in value:
            if type(value[key]) == torch.Tensor :
                value[key] = value[key].to(device)

    return data

def train_loop(model : Sam, trainloader : DataLoader, optimizer : Optimizer, epochs : int, loss_fn : callable, 
               evalloader : DataLoader = None, model_save_dir : str = None, device : str = 'cpu', verbose : bool = True, 
               use_wandb : bool = False, is_original_loss : bool = False) -> dict:
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

    for epoch in range(epochs):
        losses = []

        model.train()
        for data, mask in tqdm(trainloader, disable = not verbose, desc = f'Epoch {epoch + 1}/{epochs} - Training'):
            data = data_to_gpu(data, 'cuda')
            mask = mask.to('cuda')
    
            optimizer.zero_grad()
            pred_model, pred_iou = model(data, multimask_output = True, binary_mask_output = False)

            if is_original_loss:
                loss, ... = loss_fn(pred_model.float(), mask.float(), pred_iou.float())
            else:
                loss = loss_fn(pred_model.float(), mask.float())

            loss.backward()

            optimizer.step()
            losses.append(loss.item())
        
        mean_loss = sum(losses) / len(losses)
        if verbose:
            print(f'Loss: {mean_loss}')

        if use_wandb:
            wandb.log({"loss": mean_loss})

        if model_save_dir is not None:
            model_save_path = os.path.join(model_save_dir, 'last_model.pt')
            checkpoint_save_path = os.path.join(model_save_dir, 'last_checkpoint.pt')

            torch.save(model.state_dict(), model_save_path)
            torch.save({'epoch': epoch, 'optimizer': optimizer.state_dict()}, checkpoint_save_path)

        if evalloader is not None:
            scores = eval_loop(model, evalloader, device)
            print(f'Evaluation - BCE: {scores["BCE"]}, Dice: {scores["dice"]}, IoU: {scores["iou"]}, Precision: {scores["precision"]}, Recall: {scores["recall"]}, Time: {scores['prediction_time']}')

            if use_wandb:
                wandb.log({"bce": scores["BCE"], "dice": scores["dice"], "iou": scores["iou"], "precision": scores["precision"], "recall": scores["recall"], "time": scores['prediction_time'], "loss": sum(losses)/len(losses)})
            
            if best_loss > scores['BCE']:
                best_loss = scores['BCE'] 
                torch.save(model.state_dict(), f'{model_save_dir}/best_model.pt')

    return scores

def train_from_last_checkpoint(model : Sam, trainloader : DataLoader, optimizer : Optimizer, epochs : int, loss_fn : callable, evalloader : DataLoader = None, 
                               model_save_dir : str = None, device : str = 'cpu', verbose : bool = True, use_wandb : bool = False, last_epoch : int = 0) -> dict:
    """Function to train a model from a last checkpoint on a dataloader.
    model: nn.Module, model to train
    trainloader: DataLoader, dataloader to use for the training
    optimizer: Adam, optimizer to use for the training
    epochs: int, number of epochs to train the model
    evalloader: DataLoader, If provided, evaluate the model at each epochs on it. Default: None
    model_save_dir: str, If provided, save the model at each epochs. Also save the best model (evaluation loss) if evalloader is provided. Default: None
    device: str, device to use for the training
    last_epoch: int, last epoch trained. Default: 0
    Returns: dict, dictionary with the training metrics"""
    best_loss = float('inf')

    for epoch in range(last_epoch, epochs):
        losses = []

        for data, mask in tqdm(trainloader, disable=not verbose, desc=f'Epoch {epoch + 1}/{epochs} - Training'):
            mask = mask.to(device)

            optimizer.zero_grad()
            pred_model, _ = model(data, multimask_output = True, binary_mask_output = False)

            loss = loss_fn(pred_model.float(), mask.float())
            loss.backward()

            optimizer.step()
            losses.append(loss.item())

        if use_wandb:
            wandb.log({"loss": sum(losses)/len(losses)})

        if verbose:
            print(f'Loss: {sum(losses)/len(losses)}')

        if model_save_dir is not None:
            torch.save(model.state_dict(), f'{model_save_dir}/last_model.pt')
            torch.save({'epoch': epoch, 'optimizer': optimizer.state_dict()}, f'{model_save_dir}/last_checkpoint.pt')

        if evalloader is not None:
            scores = eval_loop(model, evalloader, device)
            print(f'Evaluation - Dice: {scores["dice"]}, IoU: {scores["iou"]}, Precision: {scores["precision"]}, Recall: {scores["recall"]}, Time: {scores['prediction_time']}')

            if use_wandb:
                wandb.log({"dice": scores["dice"], "iou": scores["iou"], "precision": scores["precision"], "recall": scores["recall"], "time": scores['prediction_time']})

            if best_loss > scores['dice']:
                best_loss = scores['dice'] 
                torch.save(model.state_dict(), f'{model_save_dir}/best_model.pt')

    return scores

if __name__ == '__main__':
    parser = ArgumentParser(description='Train a model on a dataset')
    parser.add_argument('--config', required=False, type=str, help='Path to the configuration file', default='config.toml')

    args = parser.parse_args()
    config = load_config(args.config)
    scores = train_with_config(config)

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