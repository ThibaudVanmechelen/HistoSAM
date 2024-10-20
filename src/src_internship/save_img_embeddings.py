'''Save img embeddings in file for further use. Allows to train SAM model without touching its image encoder'''
import os
from argparse import ArgumentParser

import numpy as np
import torch
from dataset_processing.dataset import SAMDataset
from model.model import TrainableSam, load_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.config import load_config


def save_img_embeddings(config:dict):
    '''Save img embeddings in file for further use. Allows to train SAM model without touching its image encoder'''
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
    img_embeddings = []
    model.eval()
    for i, (data, _) in tqdm(enumerate(dataset), total=len(dataset), desc='Saving img embeddings'):
        with torch.no_grad():
            img_embedding = model.get_image_embeddings([data])
        file_name = dataset.images[i]
        file_name = file_name.split('/')[-2]
        os.makedirs(f'{config.cytomine.dataset_path}img_embeddings/', exist_ok=True)
        torch.save(img_embedding, f'{config.cytomine.dataset_path}img_embeddings/{file_name}.pt')

def save_prompts(config:dict):
    dataset = SAMDataset(root=config.cytomine.dataset_path,
                            prompt_type={'points':True, 'box':True, 'neg_points':True, 'mask':True},
                            n_points=config.dataset.n_points,
                            n_neg_points=config.dataset.n_neg_points,
                            verbose=True,
                            to_dict=True,
                            neg_points_inside_box=config.dataset.negative_points_inside_box,
                            points_near_center=config.dataset.points_near_center,
                            random_box_shift=config.dataset.random_box_shift,
                            mask_prompt_type=config.dataset.mask_prompt_type,
                            box_around_mask=config.dataset.box_around_prompt_mask)
    prompts = dataset.prompts
    torch.save(prompts, f'{config.cytomine.dataset_path}prompts.pt')

def split_dataset(config:dict, train_ratio:float=0.8, seed:int=None):
    '''Split a dataset into train / test.
    Move splits into separate folders.'''
    files = os.listdir(config.cytomine.dataset_path + '/train/processed/')
    if seed is not None:
        np.random.seed(seed)
    suffled_files = np.random.permutation(files)
    train_files = suffled_files[:int(len(files)*(1 -train_ratio))]
    test_files = suffled_files[int(len(files)*(train_ratio)):]
    print(f'{len(train_files)} train files, {len(test_files)} test files')
    os.makedirs(config.cytomine.dataset_path + '/train/', exist_ok=True)
    os.makedirs(config.cytomine.dataset_path + '/valid/', exist_ok=True)
    #for file in train_files:
    #    os.rename(config.cytomine.dataset_path + file, config.cytomine.dataset_path + '/train/processed/' + file)
    for file in test_files:
        os.rename(config.cytomine.dataset_path + 'train/processed/' + file, config.cytomine.dataset_path + '/valid/processed/' + file)

if __name__ == '__main__':
    parser = ArgumentParser(description='Save img embeddings in file for further use. Allows to train SAM model without touching its image encoder')
    parser.add_argument('--config', required=False, type=str, help='Path to the configuration file. Default: config.toml', default='config.toml')
    args = parser.parse_args()
    config = load_config(args.config)
    #save_img_embeddings(config)
    save_prompts(config)
    #split_dataset(config)