import os
import gc
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from model.sam2_model import TrainableSAM2
from model.model import load_model
from dataset_processing.dataset import SAMDataset
from utils.config import load_config

def run_visualization(dataset_path : str, config_path : str, checkpoint_path : str, is_sam2 : bool, output_dir_path : str, file_name_format: str, last_model_path : str = None):
    print("Loading the configs")
    config = load_config(config_path)

    print("Creating the dataset...")
    prompt_type = {
        'points' : config.testing.points,
        'box' : config.testing.box,
        'neg_points' : config.testing.negative_points,
        'mask' : config.testing.mask_prompt
    }

    dataset = SAMDataset(
        root = dataset_path,
        prompt_type = prompt_type,
        n_points = config.testing.n_points,
        n_neg_points = config.testing.n_neg_points,
        verbose = True,
        to_dict = True,
        use_img_embeddings = False,
        neg_points_inside_box = config.testing.negative_points_inside_box,
        points_near_center = config.testing.points_near_center,
        random_box_shift = config.testing.random_box_shift,
        mask_prompt_type = config.testing.mask_prompt_type,
        box_around_mask = config.testing.box_around_prompt_mask,
        is_sam2_prompt = is_sam2
    )

    if not is_sam2:
        print("Starting Testing for SAM...")
        model = load_model(checkpoint_path, config.sam.model_type, img_embeddings_as_input = False, return_iou = False).to(config.misc.device)

        if last_model_path:
            print("Loading the last model...")
            model.load_state_dict(torch.load(last_model_path))

    else:
        print("Starting Testing for SAM2...")
        model = TrainableSAM2(finetuned_model_name = '', cfg = config.sam2.model_type, checkpoint = checkpoint_path, mode = "eval",
                              do_train_prompt_encoder = False, do_train_mask_decoder = False,
                              img_embeddings_as_input = False, device = config.misc.device, weight_path = last_model_path)

    test_on_sample(model, dataset, is_sam2, output_dir_path, file_name_format, nb_sample = config.testing.nb_sample, device = config.misc.device)

    del model
    del dataset
    torch.cuda.empty_cache()
    gc.collect()


def make_sample_figure(imgs, titles, save_path):
    _, axes = plt.subplots(1, len(imgs), figsize = (len(imgs) * 5, 5))

    for i, ax in enumerate(axes):
        ax.imshow(imgs[i], cmap = 'viridis' if len(imgs[i].shape) == 2 else None)
        ax.set_title(titles[i], fontsize = 12)
        ax.axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches = 'tight', dpi = 600)
    plt.close()


def test_on_sample(model, dataset : SAMDataset, is_sam2 : bool, output_dir_path : str, file_name_format : str, nb_sample : int, device : str, seed : int = 42):
    random.seed(seed)
    samples_idx = random.sample(range(len(dataset)), nb_sample)

    for i in range(nb_sample):
        sample_data, sample_mask = dataset[samples_idx[i]]
        sample_mask_np = sample_mask
        sample_mask = torch.tensor(sample_mask, dtype = torch.float32, device = device)

        original_img = np.transpose(sample_data['image'].cpu().numpy(), (1, 2, 0)).astype(np.uint8) # HxWx3
        original_mask = sample_mask_np

        if not is_sam2:
            pred = model([sample_data], multimask_output = True, binary_mask_output = True) 
            pred = pred[0].cpu().numpy() # np.ndarray HxW

            prompts = {'point_coords': None, 'point_labels': None, 'boxes': None, 'mask_inputs': None}

            prompts['boxes'] = dataset.prompts['box'][samples_idx[i]]
            prompts['mask_inputs'] = dataset.prompts['mask'][samples_idx[i]]

            point_coords = dataset.prompts['points'][samples_idx[i]]
            neg_point_coords = dataset.prompts['neg_points'][samples_idx[i]]

            if point_coords is not None and neg_point_coords is not None:
                point_labels = np.ones(len(point_coords), dtype = np.float32)
                neg_point_labels = np.zeros(len(neg_point_coords), dtype = np.float32)

                prompts['point_coords'] = np.concatenate([point_coords, neg_point_coords], axis = 0)
                prompts['point_labels'] = np.concatenate([point_labels, neg_point_labels], axis = 0)

            elif point_coords is not None:
                point_labels = np.ones(len(point_coords), dtype = np.float32)

                prompts['point_coords'] = point_coords
                prompts['point_labels'] = point_labels

            elif neg_point_coords is not None:
                neg_point_labels = np.zeros(len(neg_point_coords), dtype = np.float32)

                prompts['point_coords'] = neg_point_coords
                prompts['point_labels'] = neg_point_labels

        else:
            prompts = model.convert_prompts_from_sam_to_sam2_format(sample_data)
            sam2_img = model.convert_img_from_sam_to_sam2_format(sample_data['image'])

            model.set_image(sam2_img)

            pred_masks, pred_scores, _ = model.predict(
                point_coords = prompts.get("point_coords", None),
                point_labels = prompts.get("point_labels", None),
                box =  prompts.get("boxes", None),
                mask_input = prompts.get("mask_inputs", None),
                multimask_output = True,
                return_logits = False,
                normalize_coords = True
            )

            pred = pred_masks[np.argmax(pred_scores)] # np.ndarray HxW
    
        original_prompts = original_img.copy()
        if prompts.get("point_coords", None) is not None and prompts.get("point_labels", None):
            for (x, y), label in zip(prompts.get("point_coords", None), prompts.get("point_labels", None)):
                color = (0, 255, 0) if label == 1 else (255, 0, 0)
                cv2.circle(original_prompts, (int(x), int(y)), radius = 3, color = color, thickness = -1)

        if prompts.get("boxes", None) is not None:
            x1, y1, x2, y2 = map(int, prompts.get("boxes", None))
            cv2.rectangle(original_prompts, (x1, y1), (x2, y2), color = (0, 0, 255), thickness = 2)

        if prompts.get("mask_inputs", None) is not None:
            mask_resized = cv2.resize(prompts.get("mask_inputs", None), (original_prompts[1], original_prompts[0]), interpolation = cv2.INTER_NEAREST)
            mask_overlay = np.zeros_like(original_prompts, dtype = np.uint8)
            mask_overlay[:, :, 0] = mask_resized * 255
            original_prompts = cv2.addWeighted(original_prompts, 0.7, mask_overlay, 0.3, 0)

        save_path = os.path.join(output_dir_path, file_name_format + f'_{i}.png')
        make_sample_figure([original_img, original_mask, original_prompts, pred], ['Original Image', 'Ground Truth Mask', 'Prompts', 'Result'], save_path = save_path)