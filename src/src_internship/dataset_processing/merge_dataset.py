import os
import shutil
from random import choice
from typing import List

from tqdm import tqdm


def get_img_per_slide_count(dataset_files:List[str]) -> int:
    '''Return the number of img per slide in a dataset.'''
    img_per_slide_count = {}
    for file_path in dataset_files:
        file_name = file_path.split('/')[-1]
        img, _ = file_name.split('_')
        if img not in img_per_slide_count:
            img_per_slide_count[img] = 0
        img_per_slide_count[img] += 1
    return img_per_slide_count

def get_img_per_splits(dataset_files:List[str], splits:list=[0.6, 0.2, 0.2])->dict:
    '''Return the number of img per split in a dataset.'''
    img_per_slide_count = get_img_per_slide_count(dataset_files)
    nb_img = len(dataset_files)
    train_count = round(nb_img * splits[0])
    valid_count = round(nb_img * splits[1])
    test_count = round(nb_img * splits[2])
    counts = {'train': train_count, 'valid': valid_count, 'test': test_count}
    splits = {'train': [], 'valid': [], 'test': []}
    available_split_choice = ['train', 'valid', 'test']
    for splits_key in available_split_choice:
        if counts[splits_key] == 0:
            available_split_choice.remove(splits_key)
    while nb_img > len(splits['train']) + len(splits['valid']) + len(splits['test']):
        random_key = choice(list(img_per_slide_count.keys()))
        random_split = choice(available_split_choice)
        splits[random_split] = splits[random_split] + [int(random_key)]*img_per_slide_count[random_key]
        img_per_slide_count[random_key] = 0
        if len(splits[random_split]) >= counts[random_split]:
            available_split_choice.remove(random_split)
    return splits

def merge_datasets(root:str, datasets:List[str], verbose:bool=True) -> None:
    '''Merge n annotation_wise_scraper dataset into one. Copy them in a new directory.
    It is used to create large scale dataset for training and evaluation.
    root: str, path to the new directory where the datasets will be copied.
    datasets: List[str], list of paths to the datasets to merge.'''
    # Copy two first dataset in train
    for i, dataset_path in tqdm(enumerate((datasets[0], datasets[1])), total=len(datasets), desc='Merging datasets', disable=not verbose):
        dataset_files = get_file_path_list(dataset_path)
        splits = get_img_per_splits(dataset_files, splits=[0.8, 0.2, 0.0])
        for file_path in tqdm(dataset_files):
            file_name = file_path.split('/')[-1]
            img, _ = file_name.split('_')
            if int(img) in splits['train']:
                new_file_path = f'{root}/train/processed/{i}_{file_name}'
            elif int(img) in splits['valid']:
                new_file_path = f'{root}/valid/processed/{i}_{file_name}'
            elif int(img) in splits['test']:
                new_file_path = f'{root}/test/processed/{i}_{file_name}'
            copy_file(file_path, new_file_path)
    # Split the third dataset in train, valid and test. Make sure that img from train, valid and test comes from different whole slide images.
    dataset_path = datasets[2]
    dataset_files = get_file_path_list(dataset_path)
    splits = get_img_per_splits(dataset_files, splits=[0.64, 0.16, 0.2])
    for file_path in tqdm(dataset_files):
        file_name = file_path.split('/')[-1]
        img, _ = file_name.split('_')
        if int(img) in splits['train']:
            new_file_path = f'{root}/train/processed/2_{file_name}'
        elif int(img) in splits['valid']:
            new_file_path = f'{root}/valid/processed/2_{file_name}'
        elif int(img) in splits['test']:
            new_file_path = f'{root}/test/processed/2_{file_name}'
        copy_file(file_path, new_file_path)
    

def get_file_path_list(root:str) -> List[str]:
    '''Get the list of files in a directory.
    root: str, path to the directory
    Returns: List[str], list of file paths.'''
    return [f'{root}/{file}' for file in os.listdir(root)]

def copy_file(file_path:str, new_file_path:str) -> None:
    '''Copy a file to a new location.
    file_path: str, path to the file to copy
    new_file_path: str, path to the new location.'''
    os.makedirs(new_file_path, exist_ok=True)
    shutil.copy(file_path + '/img.jpg', new_file_path + '/img.jpg')
    shutil.copy(file_path + '/mask.jpg', new_file_path + '/mask.jpg')

if __name__ == '__main__':
    root = '../../datasets/all/'
    datasets = ['../../datasets/camelyon16.1/processed/', '../../datasets/LBTD-NEO04/processed/', '../../datasets/LBTD-AGDC10/processed/']
    #datasets = ['../../datasets/void/processed/', '../../datasets/void/processed/', '../../datasets/LBTD-AGDC10/processed/']
    merge_datasets(root, datasets)