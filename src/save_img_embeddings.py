"""Save img embeddings in file for further use. Allows to train SAM model without touching its image encoder"""
import os

import torch
from dataset_processing.dataset import SAMDataset
from model.model import load_model
from model.sam2_model import TrainableSAM2
from tqdm import tqdm

def save_img_embeddings(config : dict, dataset_path : str, checkpoint_path : str, output_dir : True, is_sam2 : bool):
    """Save img embeddings in file for further use. Allows to train SAM model without touching its image encoder"""
    os.makedirs(output_dir, exist_ok = True)

    if not is_sam2:
        model = load_model(checkpoint_path, config.sam.model_type).to(device = config.misc.device)
    else:
        model = TrainableSAM2(finetuned_model_name = "embedding_model", cfg = config.sam2.model_type, checkpoint = checkpoint_path, mode = "eval",
                              do_train_prompt_encoder = False, do_train_mask_decoder = False, img_embeddings_as_input = False, device = config.misc.device)

    dataset = SAMDataset(
        root = dataset_path,
        prompt_type = {'points' : False, 'box' : False, 'neg_points' : False, 'mask' : False},
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
    
    model.eval()
    with torch.no_grad():
        for i, (data, _) in tqdm(enumerate(dataset), total = len(dataset), desc = 'Saving img embeddings'):
            if not is_sam2:
                img_embedding = model.get_image_embeddings([data])
            else:
                sam2_image = model.convert_img_from_sam_to_sam2_format(data['image'])
                model.set_image(sam2_image)
                img_embedding = model._features

            file_name = dataset.images[i]
            file_name = file_name.split('/')[-2]

            save_path = os.path.join(output_dir, f"{file_name}.pt")
            torch.save(img_embedding, save_path)

def save_prompts(config : dict, dataset_path : str, output_path : str):
    dataset = SAMDataset(
        root = dataset_path,
        prompt_type = {'points' : True, 'box' : True, 'neg_points' : True, 'mask' : True},
        n_points = config.dataset.n_points,
        n_neg_points = config.dataset.n_neg_points,
        verbose = True,
        to_dict = True,
        neg_points_inside_box = config.dataset.negative_points_inside_box,
        points_near_center = config.dataset.points_near_center,
        random_box_shift = config.dataset.random_box_shift,
        mask_prompt_type = config.dataset.mask_prompt_type,
        box_around_mask = config.dataset.box_around_prompt_mask,
        is_sam2_prompt = False # Don't care here, won't use to_dict
    )

    torch.save(dataset.prompts, output_path)