"""Save img embeddings in file for further use. Allows to train SAM model without touching its image encoder"""
import os
from argparse import ArgumentParser

import torch
from dataset_processing.dataset import SAMDataset
from model.model import load_model
from tqdm import tqdm
from utils.config import load_config


def save_img_embeddings(config : dict):
    """Save img embeddings in file for further use. Allows to train SAM model without touching its image encoder"""
    model = load_model(config.sam.checkpoint_path, config.sam.model_type).to(device=config.misc.device)
    dataset = SAMDataset(root=config.cytomine.dataset_path,
                            prompt_type={'points':False, 'box':False, 'neg_points':False, 'mask':False},
                            n_points=config.dataset.n_points,
                            n_neg_points=config.dataset.n_neg_points,
                            verbose=True,
                            to_dict=True,
                            neg_points_inside_box=config.dataset.negative_points_inside_box,
                            points_near_center=config.dataset.points_near_center,
                            random_box_shift=config.dataset.random_box_shift,
                            mask_prompt_type=config.dataset.mask_prompt_type,
                            box_around_mask=config.dataset.box_around_prompt_mask)
    
    model.eval()

    for i, (data, _) in tqdm(enumerate(dataset), total=len(dataset), desc='Saving img embeddings'):
        with torch.no_grad():
            img_embedding = model.get_image_embeddings([data])

        file_name = dataset.images[i]
        file_name = file_name.split('/')[-2]

        os.makedirs(f'{config.cytomine.dataset_path}img_embeddings/', exist_ok=True)
        torch.save(img_embedding, f'{config.cytomine.dataset_path}img_embeddings/{file_name}.pt')

def save_prompts(config : dict):
    dataset = SAMDataset(root = config.cytomine.dataset_path,
                            prompt_type = {'points':True, 'box':True, 'neg_points':True, 'mask':True},
                            n_points = config.dataset.n_points,
                            n_neg_points = config.dataset.n_neg_points,
                            verbose = True,
                            to_dict = True,
                            neg_points_inside_box = config.dataset.negative_points_inside_box,
                            points_near_center = config.dataset.points_near_center,
                            random_box_shift = config.dataset.random_box_shift,
                            mask_prompt_type = config.dataset.mask_prompt_type,
                            box_around_mask = config.dataset.box_around_prompt_mask)

    prompts = dataset.prompts
    torch.save(prompts, f'{config.cytomine.dataset_path}prompts.pt')

if __name__ == '__main__':
    parser = ArgumentParser(description='Save img embeddings in file for further use. Allows to train SAM model without touching its image encoder')
    parser.add_argument('--config', required=False, type=str, help='Path to the configuration file. Default: config.toml', default='config.toml')
    args = parser.parse_args()
    config = load_config(args.config)
    save_img_embeddings(config)
    save_prompts(config)