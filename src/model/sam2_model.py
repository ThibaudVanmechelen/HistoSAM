from typing import List

import os
import time
import torch
import wandb
import numpy as np

from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

from cv2 import INTER_NEAREST, resize
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

from dataset_processing.dataset import (
    SAMDataset,
    AugmentedSamDataset,
    SamDatasetFromFiles,
    filter_dataset,
)
from dataset_processing.preprocess import collate_fn

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

IMG_RESOLUTION = 1024

class TrainableSAM2(SAM2ImagePredictor):
    """A trainable version of Sam2.
    Note: you could add a parameter to also train the image encoder using: self.model.image_encoder.train(True) but the corresponding file in SAM2 repository contains multiple no_grad
    instructions which need to be removed, thus i did not include it in the following code since I don't want to train the image encoder and removing those statements would slow the whole
    training down by computing unnecessary gradients.
    """
    def __init__(self, finetuned_model_name : str, cfg : str, checkpoint : str, mode : str,
                 do_train_prompt_encoder : bool, do_train_mask_decoder : bool, img_embeddings_as_input : bool, device : str):
        model = build_sam2(config_file = cfg,  ckpt_path = checkpoint, device = device, mode = mode,  apply_postprocessing = True)
        super().__init__(model)

        self.finetuned_model_name = finetuned_model_name
        self.img_embeddings_as_input = img_embeddings_as_input
        self.device = device

        if do_train_prompt_encoder:
            self.model.sam_prompt_encoder.train(do_train_prompt_encoder)

        if do_train_mask_decoder:
            self.model.sam_mask_decoder.train(do_train_mask_decoder)

    def setup_training_parameters(self, lr : float, weight_decay : float, step_size : int, gamma : float):
        self.optimizer = AdamW(params = self.model.parameters(), lr = lr, weight_decay = weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size = step_size, gamma = gamma)
        self.scaler = GradScaler()
 
    def save_embeddings(self, config : dict): # TODO make the config
        self.model.eval()

        dataset = SAMDataset(root = config.save_embeddings.dataset_path,
                             prompt_type = {'points' : False, 
                                            'box' : False, 
                                            'neg_points' : False, 
                                            'mask' : False},
                             n_points = 0,
                             n_neg_points = 0,
                             verbose = True,
                             to_dict = True,
                             neg_points_inside_box = False,
                             points_near_center = -1,
                             random_box_shift = 0,
                             mask_prompt_type = 'scribble',
                             box_around_mask = False)

        output_dir = config.save_embeddings.output_path
        os.makedirs(output_dir, exist_ok = True)

        for i, (data, _) in tqdm(enumerate(dataset), total = len(dataset), desc = 'Saving img embeddings'):
            with torch.no_grad():
                sam2_image = self.convert_img_from_sam_to_sam2_format(data['image'])
                self.set_image(sam2_image)

                features = self._features # do not need to save orig_hw because standard 1024 x 1024 which is specified in to_dict from preprocess.py

            file_name = dataset.images[i]
            file_name = file_name.split('/')[-2]

            save_path = os.path.join(output_dir, f"{file_name}.pt")
            torch.save(features, save_path)
    
    def load_weights(self, weight_path : str) -> None:
        self.model.load_state_dict(torch.load(weight_path))
        self.model.eval()

    def move_to_gpu(self, data : list[dict], device : str = 'cuda') -> list[dict]:
        """Move data to a device."""
        for value in data:
            for key in value:
                if type(value[key]) == torch.Tensor:
                    value[key] = value[key].to(device)

        return data
    
    def load_embedding_to_model(self, data : list[dict]): # to device ??????
        self._orig_hw = []
        image_embeds = []
        high_res_feats = []

        for idx in len(data):
            self._orig_hw.append(data[idx]['original_size'])

            temp_embed = data[idx]['image']['image_embed']
            temp_res = data[idx]['image']['high_res_feats']

            image_embeds.append(temp_embed)
            high_res_feats.append(temp_res)

        batched_image_embed = torch.stack(image_embeds, dim = 0).to(self.device)

        batched_high_res_feats = []
        for u in range(len(high_res_feats[0])):
            batched_feature = torch.stack([feat[u] for feat in high_res_feats], dim = 0).to(self.device)
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
    
    def train_model(self, config : dict, use_dataset : List[bool]):
        train_dataset = SamDatasetFromFiles(root = config.cytomine.dataset_path,
                                prompt_type = {'points' : config.dataset.points, 
                                               'box' : config.dataset.box, 
                                               'neg_points' : config.dataset.negative_points, 
                                               'mask' : config.dataset.mask_prompt},
                                n_points = config.dataset.n_points,
                                n_neg_points = config.dataset.n_neg_points,
                                verbose = True,
                                to_dict = True,
                                use_img_embeddings = config.training.use_img_embeddings,
                                random_box_shift = config.dataset.random_box_shift,
                                mask_prompt_type = config.dataset.mask_prompt_type,
                                box_around_mask = config.dataset.box_around_prompt_mask,
                                load_on_cpu = True,
                                filter_files = lambda x: filter_dataset(x, use_dataset)
        )

        valid_dataset = SamDatasetFromFiles(root = config.evaluate.valid_dataset_path,
                                prompt_type = {'points' : config.dataset.points, 
                                               'box' : config.dataset.box, 
                                               'neg_points' : config.dataset.negative_points, 
                                               'mask' : config.dataset.mask_prompt},
                                n_points = config.dataset.n_points,
                                n_neg_points = config.dataset.n_neg_points,
                                verbose = True,
                                to_dict = True,
                                use_img_embeddings = config.training.use_img_embeddings,
                                random_box_shift = config.dataset.random_box_shift,
                                mask_prompt_type = config.dataset.mask_prompt_type,
                                load_on_cpu = True,
                                filter_files = lambda x: filter_dataset(x, use_dataset)
        )

        if config.misc.wandb:
            wandb.init(project = config.wandb.name, config = config)

        trainloader = DataLoader(train_dataset, batch_size = config.training.batch_size, shuffle = False, collate_fn = collate_fn)
        validloader = DataLoader(valid_dataset, batch_size = config.training.batch_size, shuffle = False, collate_fn = collate_fn)

        if config.training.eval_every_epoch:
            return self.train_loop(epochs = config.training.epochs, 
                                   trainloader = trainloader,
                                   device = config.misc.device,
                                   lr = config.training.lr,
                                   weight_decay = config.training.weight_decay,
                                   step_size = config.training.step_size,
                                   gamma = config.training.gamma,
                                   evalloader = validloader,
                                   model_save_dir = config.training.model_save_dir,
                                   use_wandb = config.misc.wandb, 
                                   input_mask_eval = config.evaluate.input_mask_eval)
        else:
            return self.train_loop(epochs = config.training.epochs, 
                                   trainloader = trainloader,
                                   device = config.misc.device,
                                   lr = config.training.lr,
                                   weight_decay = config.training.weight_decay,
                                   step_size = config.training.step_size,
                                   gamma = config.training.gamma,
                                   evalloader = None,
                                   model_save_dir = config.training.model_save_dir,
                                   use_wandb = config.misc.wandb, 
                                   input_mask_eval = config.evaluate.input_mask_eval)
    
    def train_loop(self, epochs : int, trainloader : DataLoader, device : str, lr : float = 0.0001, weight_decay : float = 0.0001, step_size : int = 500, gamma : float = 0.2, verbose : bool = True,
                    evalloader : DataLoader = None, model_save_dir : str = None,  use_wandb : bool = False, input_mask_eval : bool = False) -> dict:
        """https://www.datacamp.com/tutorial/sam2-fine-tuning"""
        self.setup_training_parameters(lr = lr, weight_decay = weight_decay, step_size = step_size, gamma = gamma)
        best_loss = float('inf')

        for epoch in range(epochs):
            epoch_losses = []
            epoch_segmentation_losses = []
            epoch_score_losses = []
            epoch_mean_ious = []

            for data, mask in tqdm(trainloader, disable = not verbose, desc = f'Epoch {epoch + 1}/{epochs} - Training'):
                data = self.move_to_gpu(data, device)
                mask = mask.to(device)

                with autocast(): # modifying floating point precision to speed up the training
                    if self.img_embeddings_as_input:
                        self.reset_predictor()
                        
                        self._is_image_set = True
                        self._is_batch = True

                        self.load_embedding_to_model(data)
                            
                    else:
                        sam2_image_batch = [self.convert_img_from_sam_to_sam2_format(x['image']) for x in data]
                        self.set_image_batch(sam2_image_batch)

                    batch_segmentation_loss = []
                    batch_score_loss = []
                    batch_ious = []

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

                        prediction_masks = self._transforms.postprocess_masks(low_res_masks, self._orig_hw[img_idx])

                        truth_mask = mask[img_idx].float()
                        segmentation_loss = BCEWithLogitsLoss(prediction_masks[:, 0], truth_mask)

                        pred_mask = torch.sigmoid(prediction_masks[:, 0]) # converting logits to probabilities for the first mask (the most probable)
                        inter = (truth_mask * (pred_mask > 0.5)).sum(1).sum(1)
                        iou = inter / (truth_mask.sum(1).sum(1) + (pred_mask > 0.5).sum(1).sum(1) - inter)
                        score_loss = torch.abs(iou_predictions[:, 0] - iou).mean()

                        batch_segmentation_loss.append(segmentation_loss)
                        batch_score_loss.append(score_loss)
                        batch_ious.append(iou.mean().item())


                    mean_segmentation_loss = torch.mean(torch.stack(batch_segmentation_loss))
                    mean_score_loss = torch.mean(torch.stack(batch_score_loss))
                    mean_iou = np.mean(batch_ious) # because does not need to be in the computation graph

                    loss = mean_segmentation_loss + 0.05 * mean_score_loss
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    self.model.zero_grad() # or self.optimizer.zero_grad() ? 
                    self.scheduler.step()

                    epoch_losses.append(loss.item())
                    epoch_segmentation_losses.append(mean_segmentation_loss.item())
                    epoch_score_losses.append(mean_score_loss.item())
                    epoch_mean_ious.append(mean_iou)

            if verbose:
                print(f'Total Loss: {sum(epoch_losses)/len(epoch_losses)}, BCE Loss: {sum(epoch_segmentation_losses)/len(epoch_segmentation_losses)}, Score Loss: {sum(epoch_score_losses)/len(epoch_score_losses)}, IoU: {sum(epoch_mean_ious)/len(epoch_mean_ious)}')

            if model_save_dir is not None:
                name_model = self.finetuned_model_name + '_last_model.pt'
                name_model = os.path.join(model_save_dir, name_model)

                torch.save(self.model.state_dict(), name_model)

                name_checkpoint = self.finetuned_model_name + '_last_checkpoint.pt'
                name_checkpoint = os.path.join(model_save_dir, name_checkpoint)
            
                torch.save({'epoch': epoch, 'optimizer': self.optimizer.state_dict()}, name_checkpoint)

            if evalloader is not None:
                scores = self.eval_loop(evalloader, input_mask_eval, return_mean = True)
                print(f'Evaluation - Total Loss: {scores["total_loss"]}, BCE Loss: {scores["BCE"]}, Score Loss: {scores["score_loss"]}, Dice: {scores["dice"]}, IoU: {scores["iou"]}, Precision: {scores["precision"]}, Recall: {scores["recall"]}, Time: {scores["prediction_time"]}')

                if use_wandb:
                    wandb.log({
                        "Total Loss": scores["total_loss"],
                        "BCE Loss": scores["BCE"], 
                        "Score Loss": scores["score_loss"],
                        "Dice": scores["dice"], 
                        "Iou": scores["iou"], 
                        "Precision": scores["precision"], 
                        "Recall": scores["recall"], 
                        "Time": scores['prediction_time']
                        })
                
                if best_loss > scores['total_loss']:
                    best_loss = scores['total_loss']

                    name_best = self.finetuned_model_name + '_best_model.pt'
                    name_best = os.path.join(model_save_dir, name_best)

                    torch.save(self.model.state_dict(), name_best)

            elif use_wandb:
                wandb.log({
                    "Total Loss": sum(epoch_losses)/len(epoch_losses),
                    "BCE Loss": sum(epoch_segmentation_losses)/len(epoch_segmentation_losses),
                    "Score Loss": sum(epoch_score_losses)/len(epoch_score_losses), 
                    "IoU": sum(epoch_mean_ious)/len(epoch_mean_ious)
                    })
                
        return scores
    
    def evaluate_model(self, config : dict, use_dataset : List[bool] = [True, True, True]):
        test_dataset = SamDatasetFromFiles(root = config.evaluate.test_dataset_path,
                                prompt_type = {'points' : config.dataset.points, 
                                               'box' : config.dataset.box, 
                                               'neg_points' : config.dataset.negative_points, 
                                               'mask' : config.dataset.mask_prompt},
                                n_points = config.dataset.n_points,
                                n_neg_points = config.dataset.n_neg_points,
                                verbose = True,
                                to_dict = True,
                                use_img_embeddings = config.training.use_img_embeddings,
                                random_box_shift = config.dataset.random_box_shift,
                                mask_prompt_type = config.dataset.mask_prompt_type,
                                box_around_mask = config.dataset.box_around_prompt_mask,
                                load_on_cpu = True,
                                filter_files = lambda x: filter_dataset(x, use_dataset)
        )

        dataloader = DataLoader(test_dataset, batch_size = config.training.batch_size, shuffle = False, collate_fn = collate_fn)
        scores = self.eval_loop(dataloader = dataloader, 
                                input_mask_eval = config.evaluate.input_mask_eval)

        return scores
    
    def eval_loop(self, dataloader : DataLoader, input_mask_eval : bool) -> dict:
        scores = {
                'Total Loss':[], 'BCE Loss':[], 'Score Loss':[],
                'prediction_time':[],
                'dice':[], 'iou':[], 'precision':[], 'recall':[], 
                'dice_input':[], 'iou_input':[], 'precision_input':[], 'recall_input':[]
                }

        self.model.eval()

        with torch.no_grad():
            for data, mask in tqdm(dataloader):
                mask = mask.to(device = self.device, dtype = torch.float32)
    
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

                pred_masks, scores, _ = self.predict_batch(
                    point_coords_batch = prompts_record.get("point_coords", None),
                    point_labels_batch = prompts_record.get("point_labels", None),
                    box_batch =  prompts_record.get("boxes", None),
                    mask_input_batch = prompts_record.get("mask_inputs", None),
                    multimask_output = True,
                    return_logits = True,
                    normalize_coords = True
                ) # pred_masks are logit masks
                end_time = time.time()

                assert len(pred_masks) == len(data), "There should be a prediction for each element in the batch."

                pred_masks = torch.stack([torch.tensor(pred, device = self.device) for pred in pred_masks])  # Shape: (N, C, H, W)
                scores = torch.stack([torch.tensor(score, device = self.device) for score in scores])  # Shape: (N, C)
                
                most_probable_mask_idx = scores.argmax(dim = 1)  # Shape: (N)
                selected_scores = scores.gather(1, most_probable_mask_idx.unsqueeze(1)).squeeze(1)  # Shape: (N), unsqueeze is to add a dim because it is needed by gather, gather 1 is to select accros dim 1, and squeeze is to remove the , 1 at the end
                selected_masks = pred_masks[torch.arange(pred_masks.size(0)), most_probable_mask_idx]  # Shape: (N, H, W), extracting the masks

                seg_loss = BCEWithLogitsLoss(selected_masks, mask.float())

                sig_masks = torch.sigmoid(selected_masks)
                binary_masks = (sig_masks > 0.5).float() # TODO check if value is indeed 0.5, github says 0.0

                intersection = (mask * binary_masks).sum(dim = (1, 2))  # Shape: (N,)
                union = mask.sum(dim = (1, 2)) + binary_masks.sum(dim = (1, 2)) - intersection  # Shape: (N,)
                iou = intersection / (union + 1e-6)  # Avoid division by zero

                # Compute score loss
                score_loss = torch.abs(selected_scores - iou).mean()


        #         for i in range(len(pred_masks)): # taking most probable mask and iou
        #             most_probable_mask_idx = scores[i].argmax()

        #             selected_score = scores[i][most_probable_mask_idx]
        #             selected_mask = pred_masks[i][most_probable_mask_idx]

        #             truth_mask = mask[i].float()
        #             seg_loss = BCEWithLogitsLoss(selected_mask.float(), truth_mask)
        #             batch_segmentation_loss.append(seg_loss)

        #             sig_mask = torch.sigmoid(selected_mask)
        #             inter = (truth_mask * (sig_mask > 0.5)).sum(1).sum(1)
        #             iou = inter / (truth_mask.sum(1).sum(1) + (sig_mask > 0.5).sum(1).sum(1) - inter)
        #             score_loss = torch.abs(selected_score - iou).mean()

        #             batch_score_loss.append(score_loss)

        #             selected_mask = np.array(selected_mask, dtype = np.uint8)
        #             mask_i = np.array(mask[i], dtype = np.uint8)

        #             y_pred = selected_mask.flatten()
        #             y_true = mask_i.flatten()
                    
        #             scores['dice'].append(f1_score(y_true, y_pred))
        #             scores['iou'].append(jaccard_score(y_true, y_pred))
        #             scores['precision'].append(precision_score(y_true, y_pred, zero_division = 1))
        #             scores['recall'].append(recall_score(y_true, y_pred, zero_division = 1))

        #             if input_mask_eval:
        #                 input_mask = resize(data[i]['mask_inputs'].cpu().numpy()[0][0], (IMG_RESOLUTION, IMG_RESOLUTION), interpolation = INTER_NEAREST)
        #                 y_input = input_mask.flatten()

        #                 scores['dice_input'].append(f1_score(y_true, y_input)) 
        #                 scores['iou_input'].append(jaccard_score(y_true, y_input))
        #                 scores['precision_input'].append(precision_score(y_true, y_input, zero_division = 1))
        #                 scores['recall_input'].append(recall_score(y_true, y_input, zero_division = 1))

        #         mean_segmentation_loss = torch.mean(torch.stack(batch_segmentation_loss))
        #         mean_score_loss = torch.mean(torch.stack(batch_score_loss))
        #         total_loss = mean_segmentation_loss + 0.05 * mean_score_loss

        #         scores['Total Loss'].append(total_loss.item())
        #         scores['BCE Loss'].append(mean_segmentation_loss.item())
        #         scores['Score Loss'].append(mean_score_loss.item())
        #         scores['prediction_time'].append(end_time - start_time)

        # return scores

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
                mask = mask.to(device = self.device, dtype = torch.float32)
    
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

                pred_masks, scores, _ = self.predict_batch(
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
                    input_masks = torch.stack([torch.from_numpy(d['mask_inputs']).to(self.device).squeeze(0) for d in data])  # Shape: (N, H, W)
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