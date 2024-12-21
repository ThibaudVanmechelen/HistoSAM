import os
import time
import torch
import wandb
import numpy as np

from tqdm import tqdm
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset_processing.dataset import (
    SamDatasetFromFiles
)
from dataset_processing.preprocess import collate_fn

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.loss import SAM_Loss, Custom_SAM2_Loss

IMG_RESOLUTION = 1024

class TrainableSAM2(SAM2ImagePredictor):
    """A trainable version of Sam2.
    Note: you could add a parameter to also train the image encoder using: self.model.image_encoder.train(True) but the corresponding file in SAM2 repository contains multiple no_grad
    instructions which need to be removed, thus i did not include it in the following code since I don't want to train the image encoder and removing those statements would slow the whole
    training down by computing unnecessary gradients.
    """
    def __init__(self, finetuned_model_name : str, cfg : str, checkpoint : str, mode : str,
                 do_train_prompt_encoder : bool, do_train_mask_decoder : bool, img_embeddings_as_input : bool, device : str, weight_path : str = None):
        model = build_sam2(config_file = cfg,  ckpt_path = checkpoint, device = device, mode = mode,  apply_postprocessing = True) # Note: mode is only really useful when using eval, otherwise does not do anything: https://github.com/facebookresearch/sam2/blob/main/sam2/build_sam.py
        super().__init__(model)

        self.finetuned_model_name = finetuned_model_name
        self.img_embeddings_as_input = img_embeddings_as_input
        self.running_device = device

        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path))

        self.do_train_prompt_encoder = do_train_prompt_encoder
        self.do_train_mask_decoder = do_train_mask_decoder

        if do_train_prompt_encoder:
            self.model.sam_prompt_encoder.train(True)

        if do_train_mask_decoder:
            self.model.sam_mask_decoder.train(True)

    def setup_training_parameters(self, lr : float, weight_decay : float, step_size : int, gamma : float, batch_size : int):
        self.optimizer = AdamW(params = self.model.parameters(), lr = lr, weight_decay = weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size = step_size // batch_size, gamma = gamma)
        self.scaler = GradScaler()

        print(f'Optimizer parameters: lr = {lr}, weight_decay = {weight_decay}')
        print(f'Scheduler parameters: step_size = {step_size // batch_size}, gamma = {gamma}')

    def move_to_gpu(self, data : list[dict], device : str = 'cuda') -> list[dict]:
        """Move data to a device."""
        for value in data:
            for key in value:
                if type(value[key]) == torch.Tensor:
                    value[key] = value[key].to(device)

        return data
    
    def load_embedding_to_model(self, data : list[dict]):
        self._orig_hw = []
        image_embeds = []
        high_res_feats = []

        for idx in len(data):
            self._orig_hw.append(data[idx]['original_size'])

            temp_embed = data[idx]['image']['image_embed']
            temp_res = data[idx]['image']['high_res_feats']

            image_embeds.append(temp_embed)
            high_res_feats.append(temp_res)

        batched_image_embed = torch.stack(image_embeds, dim = 0).to(self.running_device)

        batched_high_res_feats = []
        for u in range(len(high_res_feats[0])):
            batched_feature = torch.stack([feat[u] for feat in high_res_feats], dim = 0).to(self.running_device)
            batched_high_res_feats.append(batched_feature)
    
        self._features = {"image_embed": batched_image_embed, "high_res_feats": batched_high_res_feats}
    
    def convert_img_from_sam_to_sam2_format(self, sam_image : torch.Tensor) -> np.ndarray:
        img_np = sam_image.cpu().numpy() if sam_image.device.type == "cuda" else sam_image.numpy() # image is 3xHxW
        img_np = np.transpose(img_np, (1, 2, 0)) # image becomes HxWx3
        
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        return img_np
    
    def convert_prompts_from_sam_to_sam2_format(self, record : dict) -> dict: # Note: kind of legacy code, changed it multiple times, now more than a place holder than anything
        sam2_prompts = {'point_coords': None, 'point_labels': None, 'boxes': None, 'mask_inputs': None}
        
        sam2_prompts['point_coords'] = record.get('point_coords', None)
        sam2_prompts['point_labels'] = record.get('point_labels', None)
        sam2_prompts['boxes'] = record.get('boxes', None)
        sam2_prompts['mask_inputs'] = record.get('mask_inputs', None)

        return sam2_prompts
        
    def convert_batch_prompts_from_sam_to_sam2_format(self, data : list[dict]) -> dict:
        sam2_prompts = {'point_coords': [], 'point_labels': [], 'boxes': [], 'mask_inputs': []}

        for item in data:
            sam2_prompts['point_coords'].append(item.get('point_coords', None))
            sam2_prompts['point_labels'].append(item.get('point_labels', None))
            sam2_prompts['boxes'].append(item.get('boxes', None))
            sam2_prompts['mask_inputs'].append(item.get('mask_inputs', None))

        for key in sam2_prompts:
            if all(value is None for value in sam2_prompts[key]):
                sam2_prompts[key] = None

        return sam2_prompts
    
    def train_model(self, config : dict, training_dataset_path : str, validation_dataset_path : str, use_original_sam_loss : bool):
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")

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
            is_sam2_prompt = True,
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
                is_sam2_prompt = True,
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
            loss_fn = SAM_Loss(is_sam2_loss = True)
        else:
            loss_fn = Custom_SAM2_Loss()

        self.setup_training_parameters(lr = config.training.lr, weight_decay = config.training.weight_decay, 
                                       step_size = config.training.step_size, gamma = config.training.gamma, 
                                       batch_size = config.training.batch_size)

        if config.training.train_from_last_checkpoint:
            checkpoint = torch.load(config.training.last_checkpoint_path)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

            return self.train_loop(trainloader, config.training.epochs, loss_fn, validloader, config.training.model_save_dir, 
                                            use_wandb = config.misc.wandb, last_epoch = checkpoint['epoch'], 
                                            eval_frequency = config.validation.frequency, is_original_loss = use_original_sam_loss)
        
        return self.train_loop(trainloader, config.training.epochs, loss_fn, validloader, config.training.model_save_dir, 
                        use_wandb = config.misc.wandb, last_epoch = -1, eval_frequency = config.validation.frequency, is_original_loss = use_original_sam_loss)
    
    def train_loop(self, trainloader : DataLoader, epochs : int, loss_fn : callable, evalloader : DataLoader = None, 
                   model_save_dir : str = None, verbose : bool = True, use_wandb : bool = False, last_epoch : int = -1, 
                   eval_frequency : int = 1, is_original_loss : bool = False) -> dict:
        """https://www.datacamp.com/tutorial/sam2-fine-tuning"""
        best_loss = float('inf')

        scores = {
            'training_total_loss': [], 'training_focal_loss': [], 'training_dice_loss': [], 'training_iou_loss': [], 'training_seg_loss': [], 'training_score_loss': [],
            'validation_total_loss': [], 'validation_focal_loss': [], 'validation_dice_loss': [], 'validation_iou_loss': [], 'validation_seg_loss': [], 'validation_score_loss': [],
            'validation_dice': [], 'validation_iou': [], 'validation_precision': [], 'validation_recall': [], 'validation_prediction_time': []
        }

        start_epoch = last_epoch + 1 if last_epoch >= 0 else 0
        for epoch in range(start_epoch, epochs):
            total_losses = []

            if is_original_loss:
                focal_losses = []
                dice_losses = []
                iou_losses = []

            else:
                segmentation_losses = []
                score_losses = []

            for data, mask in tqdm(trainloader, disable = not verbose, desc = f'Epoch {epoch + 1}/{epochs} - Training'):
                data = self.move_to_gpu(data, self.running_device)
                mask = mask.to(self.running_device)

                with autocast(): # modifying floating point precision to speed up the training
                    if self.img_embeddings_as_input:
                        self.reset_predictor()
                        
                        self._is_image_set = True
                        self._is_batch = True

                        self.load_embedding_to_model(data)
                            
                    else:
                        sam2_image_batch = [self.convert_img_from_sam_to_sam2_format(x['image']) for x in data]
                        self.set_image_batch(sam2_image_batch) # this function does not compute grad for the img encoder

                    batch_best_pred = []
                    batch_best_ious = []

                    num_images = len(self._features['image_embed'])
                    for img_idx in range(num_images):
                        image_record = data[img_idx]
                        converted_prompts = self.convert_prompts_from_sam_to_sam2_format(image_record)

                        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                            converted_prompts.get('point_coords', None), 
                            converted_prompts.get('point_labels', None), 
                            box = converted_prompts.get('boxes', None), 
                            mask_logits = converted_prompts.get('mask_inputs', None), 
                            normalize_coords = True,
                            img_idx = img_idx)
                        
                        if unnorm_coords is not None:
                            concat_points = (unnorm_coords, labels)
                        else:
                            concat_points = None

                        if unnorm_box is not None:
                            box_coords = unnorm_box.reshape(-1, 2, 2)

                            box_labels = torch.tensor([[2, 3]], dtype = torch.int, device = unnorm_box.device)
                            box_labels = box_labels.repeat(unnorm_box.size(0), 1)

                            if concat_points is not None:
                                concat_coords = torch.cat([box_coords, concat_points[0]], dim = 1)
                                concat_labels = torch.cat([box_labels, concat_points[1]], dim = 1)

                                concat_points = (concat_coords, concat_labels)
                            else:
                                concat_points = (box_coords, box_labels)

                        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(points = concat_points, boxes = None, masks = mask_input,)

                        batched_mode = (concat_points is not None and concat_points[0].shape[0] > 1)
                        high_res_features = [feat_level[img_idx].unsqueeze(0) for feat_level in self._features["high_res_feats"]]

                        low_res_masks, iou_predictions, _ , _ = self.model.sam_mask_decoder(
                            image_embeddings = self._features["image_embed"][img_idx].unsqueeze(0),
                            image_pe = self.model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings = sparse_embeddings,
                            dense_prompt_embeddings = dense_embeddings,
                            multimask_output = True,
                            repeat_image = batched_mode,
                            high_res_features = high_res_features,
                        )

                        prediction_masks = self._transforms.postprocess_masks(low_res_masks, self._orig_hw[img_idx]) # Shape: (B, C, H, W), here B is just 1 TODO check this dim
                        prediction_masks = prediction_masks.squeeze(0) # Shape: (C, H, W)
                        iou_predictions = iou_predictions.squeeze(0) # Shape: C

                        most_probable_mask_idx = iou_predictions.argmax()
                        prediction_masks = prediction_masks[most_probable_mask_idx, :, :]
                        prediction_iou = iou_predictions[most_probable_mask_idx]

                        batch_best_pred.append(prediction_masks)
                        batch_best_ious.append(prediction_iou)

                    pred_masks = torch.stack(batch_best_pred) # Shape: (N, H, W)
                    pred_ious = torch.stack(batch_best_ious) # Shape: (N, 1) TODO check this 2 dims

                    if is_original_loss:
                        loss, loss_parts = loss_fn(pred_masks.float(), mask.float(), pred_ious.float())

                        focal_losses.append(loss_parts['focal'])
                        dice_losses.append(loss_parts['dice'])
                        iou_losses.append(loss_parts['iou'])

                    else:
                        loss, loss_parts = loss_fn(pred_masks.float(), mask.float(), pred_ious.float(), self.mask_threshold)

                        segmentation_losses.append(loss_parts['seg'])
                        score_losses.append(loss_parts['score'])

                    total_losses.append(loss.item())

                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    self.model.zero_grad()
                    self.scheduler.step()

            mean_total_loss = sum(total_losses) / len(total_losses)
            scores['training_total_loss'].append(mean_total_loss)

            if is_original_loss:
                mean_focal_loss = sum(focal_losses) / len(focal_losses)
                mean_dice_loss = sum(dice_losses) / len(dice_losses)
                mean_iou_loss = sum(iou_losses) / len(iou_losses)

                scores['training_focal_loss'].append(mean_focal_loss)
                scores['training_dice_loss'].append(mean_dice_loss)
                scores['training_iou_loss'].append(mean_iou_loss)
            else:
                mean_seg_loss = sum(segmentation_losses) / len(segmentation_losses)
                mean_score_loss = sum(score_losses) / len(score_losses)

                scores['training_seg_loss'].append(mean_seg_loss)
                scores['training_score_loss'].append(mean_score_loss)
            
            if verbose:
                if is_original_loss:
                    print(f'Training - Mean Total Loss: {mean_total_loss}, Focal Loss: {mean_focal_loss}, Dice Loss: {mean_dice_loss}, IoU: {mean_iou_loss}')
                else:
                    print(f'Training - Mean Total Loss: {mean_total_loss}, Segmentation Loss: {mean_seg_loss}, Score Loss: {mean_score_loss}')

            if use_wandb:
                if is_original_loss:
                    wandb.log({'total_loss': mean_total_loss, 'focal_loss': mean_focal_loss, 'dice_loss': mean_dice_loss, 'iou_loss': mean_iou_loss})
                else:
                    wandb.log({'total_loss': mean_total_loss, 'seg_loss': mean_seg_loss, 'score_loss': mean_score_loss})

            if model_save_dir is not None:
                name_model = self.finetuned_model_name + '_last_model.pt'
                name_model = os.path.join(model_save_dir, name_model)

                torch.save(self.model.state_dict(), name_model)

                name_checkpoint = self.finetuned_model_name + '_last_checkpoint.pt'
                name_checkpoint = os.path.join(model_save_dir, name_checkpoint)
            
                torch.save({
                    'epoch': epoch, 
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'scaler': self.scaler.state_dict()
                }, name_checkpoint)

            if evalloader is not None and epoch % eval_frequency == 0:
                scores_eval = self.eval_loop(evalloader, input_mask_eval = False, is_original_loss = is_original_loss)
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
                        wandb.log({ "eval_total_loss": scores_eval["total_loss"],
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
                                            Seg Loss: {scores_eval["seg_loss"]},
                                            Score Loss: {scores_eval["score_loss"]},
                                            Dice: {scores_eval["dice"]}, 
                                            IoU: {scores_eval["iou"]}, 
                                            Precision: {scores_eval["precision"]}, 
                                            Recall: {scores_eval["recall"]}, 
                                            Time: {scores_eval["prediction_time"]}''')

                    if use_wandb:
                        wandb.log({ "eval_total_loss": scores_eval["total_loss"],
                                    "eval_seg_loss": scores_eval["seg_loss"],
                                    "eval_score_loss": scores_eval["score_loss"],
                                    "eval_dice": scores_eval["dice"],
                                    "eval_iou": scores_eval["iou"],
                                    "eval_precision": scores_eval["precision"], 
                                    "eval_recall": scores_eval["recall"], 
                                    "eval_time": scores_eval["prediction_time"]})
                        
                    scores['validation_total_loss'].append(scores_eval["total_loss"])
                    scores['validation_seg_loss'].append(scores_eval["seg_loss"])
                    scores['validation_score_loss'].append(scores_eval["score_loss"])
                    scores['validation_dice'].append(scores_eval["dice"])
                    scores['validation_iou'].append(scores_eval["iou"])
                    scores['validation_precision'].append(scores_eval["precision"])
                    scores['validation_recall'].append(scores_eval["recall"])
                    scores['validation_prediction_time'].append(scores_eval["prediction_time"])

                if best_loss > scores_eval["total_loss"]:
                    best_loss = scores_eval["total_loss"]
                    best_model_save_path = os.path.join(model_save_dir, 'best_model.pt')
                    
                    torch.save(self.model.state_dict(), best_model_save_path)
                
        return scores
    
    def eval_loop(self, dataloader : DataLoader, input_mask_eval : bool, is_original_loss : bool = False) -> dict:
        scores = {
            'total_loss': [], 'focal_loss': [], 'dice_loss': [], 'iou_loss': [], 'seg_loss': [], 'score_loss': [],
            'dice': [], 'iou': [], 'precision': [], 'recall': [], 'prediction_time': []
        }

        self.model.eval()

        if is_original_loss:
            loss_fn = SAM_Loss(is_sam2_loss = True)
        else:
            loss_fn = Custom_SAM2_Loss()

        with torch.no_grad():
            for data, mask in tqdm(dataloader):
                mask = mask.to(device = self.running_device, dtype = torch.float32)
    
                prompts_record = self.convert_batch_prompts_from_sam_to_sam2_format(data)
                sam2_image_batch = [self.convert_img_from_sam_to_sam2_format(x['image']) for x in data]

                start_time = time.time()
                if self.img_embeddings_as_input:
                    self.reset_predictor()
                                            
                    self._is_image_set = True
                    self._is_batch = True

                    self.load_embedding_to_model(data)
                            
                else:
                    self.set_image_batch(sam2_image_batch)

                pred_masks, pred_scores, _ = self.predict_batch(
                    point_coords_batch = prompts_record.get("point_coords", None),
                    point_labels_batch = prompts_record.get("point_labels", None),
                    box_batch =  prompts_record.get("boxes", None),
                    mask_input_batch = prompts_record.get("mask_inputs", None),
                    multimask_output = True,
                    return_logits = True,
                    normalize_coords = True
                ) # pred_masks are binary masks
                end_time = time.time()

                assert len(pred_masks) == len(data), "There should be a prediction for each element in the batch."

                pred_masks = torch.stack([torch.tensor(pred, device = self.running_device) for pred in pred_masks])  # Shape: (N, C, H, W)
                pred_scores = torch.stack([torch.tensor(score, device = self.running_device) for score in pred_scores])  # Shape: (N, C)

                most_probable_mask_idx = pred_scores.argmax(dim = 1)  # Shape: (N)
                selected_masks = pred_masks[torch.arange(pred_masks.size(0)), most_probable_mask_idx, :, :]  # Shape: (N, H, W), extracting the masks
                selected_scores = pred_scores[torch.arange(pred_scores.size(0)), most_probable_mask_idx].unsqueeze(1) # Shape: (N, 1) TODO check this dimension

                if is_original_loss:
                    loss, loss_parts = loss_fn(selected_masks.float(), mask.float(), selected_scores.float())

                    scores['focal_loss'].append(loss_parts['focal'])
                    scores['dice_loss'].append(loss_parts['dice'])
                    scores['iou_loss'].append(loss_parts['iou'])

                else:
                    loss, loss_parts = loss_fn(selected_masks.float(), mask.float(), selected_scores.float(), self.mask_threshold)

                    scores['seg_loss'].append(loss_parts['seg'])
                    scores['score_loss'].append(loss_parts['score'])

                scores['total_loss'].append(loss.item())
                scores['prediction_time'].append(end_time - start_time)

                selected_masks = torch.where(selected_masks > self.mask_threshold, 1, 0).float() # to obtain binary mask for the metrics

                y_true_flat = mask.view(mask.size(0), -1)  # Shape: (N, H * W)
                y_pred_flat = selected_masks.view(selected_masks.size(0), -1)  # Shape: same

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
                    input_masks = torch.stack([torch.from_numpy(d['mask_inputs']).to(self.running_device).squeeze(0) for d in data])  # Shape: (N, H, W)
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

    def test_loop(self, dataloader : DataLoader, input_mask_eval : bool) -> dict:
        print("### Starting testing with SAM2 ! ###")
        scores = {
                'prediction_time':[],
                'dice':[], 'iou':[], 'precision':[], 'recall':[], 
                'dice_input':[], 'iou_input':[], 'precision_input':[], 'recall_input':[]
                }

        self.model.eval()

        with torch.no_grad():
            for data, mask in tqdm(dataloader):
                mask = mask.to(device = self.running_device, dtype = torch.float32)
    
                prompts_record = self.convert_batch_prompts_from_sam_to_sam2_format(data)
                sam2_image_batch = [self.convert_img_from_sam_to_sam2_format(x['image']) for x in data]

                start_time = time.time()
                if self.img_embeddings_as_input:
                    self.reset_predictor()
                                            
                    self._is_image_set = True
                    self._is_batch = True

                    self.load_embedding_to_model(data)
                            
                else:
                    self.set_image_batch(sam2_image_batch)

                pred_masks, pred_scores, _ = self.predict_batch(
                    point_coords_batch = prompts_record.get("point_coords", None),
                    point_labels_batch = prompts_record.get("point_labels", None),
                    box_batch =  prompts_record.get("boxes", None),
                    mask_input_batch = prompts_record.get("mask_inputs", None),
                    multimask_output = True,
                    return_logits = False,
                    normalize_coords = True
                ) # pred_masks are binary masks
                end_time = time.time()

                assert len(pred_masks) == len(data), "There should be a prediction for each element in the batch."

                pred_masks = torch.stack([torch.tensor(pred, device = self.running_device) for pred in pred_masks])  # Shape: (N, C, H, W)
                pred_scores = torch.stack([torch.tensor(score, device = self.running_device) for score in pred_scores])  # Shape: (N, C)
                
                most_probable_mask_idx = pred_scores.argmax(dim = 1)  # Shape: (N)
                selected_masks = pred_masks[torch.arange(pred_masks.size(0)), most_probable_mask_idx, :, :]  # Shape: (N, H, W), extracting the masks

                scores['prediction_time'].append(end_time - start_time)
                y_true_flat = mask.view(mask.size(0), -1)  # Shape: (N, H * W)
                y_pred_flat = selected_masks.view(selected_masks.size(0), -1)  # Shape: same

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
                    input_masks = torch.stack([torch.from_numpy(d['mask_inputs']).to(self.running_device).squeeze(0) for d in data])  # Shape: (N, H, W)
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