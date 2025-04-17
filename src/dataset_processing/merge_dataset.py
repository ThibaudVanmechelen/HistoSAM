import os
import shutil

from random import choice
from typing import List

from tqdm import tqdm

def get_img_per_slide_count(dataset_files : List[str]) -> int:
    """Return the number of img per slide in a dataset.
    Basically, count the number of folder {i}_{a} from annotation_wise_scraper
    for all possible i."""
    img_per_slide_count = {}

    for file_path in dataset_files:
        file_name = file_path.split('/')[-1]
        img, _ = file_name.split('_')

        if img not in img_per_slide_count:
            img_per_slide_count[img] = 0

        img_per_slide_count[img] += 1

    return img_per_slide_count


def get_img_per_splits(dataset_files : List[str], splits : list = [0.6, 0.2, 0.2]) -> dict: # Note that the splitting here is approximate and not exact
    """Return the number of img per split in a dataset."""
    img_per_slide_count = get_img_per_slide_count(dataset_files)
    nb_img = len(dataset_files)

    train_count = round(nb_img * splits[0])
    valid_count = round(nb_img * splits[1])
    test_count = round(nb_img * splits[2])

    counts = {'train': train_count, 'valid': valid_count, 'test': test_count}
    splits = {'train': [], 'valid': [], 'test': []}

    available_split_choice = ['train', 'valid', 'test']
    for splits_key in available_split_choice:
        if counts[splits_key] <= 0:
            available_split_choice.remove(splits_key)
    
    while nb_img > len(splits['train']) + len(splits['valid']) + len(splits['test']):
        random_key = choice(list(img_per_slide_count.keys()))
        random_split = choice(available_split_choice)

        splits[random_split] = splits[random_split] + [int(random_key)] * img_per_slide_count[random_key] # repeat the id of the image (repetition = nb of annnotation)
        del img_per_slide_count[random_key]

        if len(splits[random_split]) >= counts[random_split]:
            available_split_choice.remove(random_split)

    return splits


def merge_datasets_with_different_splits(root : str, datasets : List[str], 
                                         datasets_splits : List[List[float]], verbose : bool = True) -> None: # for generalisation
    # Note validation set should come from the same datasets use for the training set in order to respoect data distribution, it must not come from the test set.
    """Merge n annotation_wise_scraper dataset into one. Copy them in a new directory.
    It is used to create large scale dataset for training and evaluation.
    In this function each of the datasets can have a different split between train, val and testing."""
    # Copy two first dataset in train and valid
    for i, dataset_path in tqdm(enumerate(datasets), total = len(datasets), desc = 'Merging datasets', disable = not verbose):
        dataset_files = get_file_path_list(dataset_path)
        dataset_i_splits = datasets_splits[i]

        splits = get_img_per_splits(dataset_files, splits = dataset_i_splits)

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

    if verbose:
        train_count = 0
        valid_count = 0
        test_count = 0

        directory = f'{root}/train/processed/'
        if os.path.exists(directory):
            train_count = count_folders(directory)
            print(f"Number of files in train: {train_count}")

        directory = f'{root}/valid/processed/'
        if os.path.exists(directory):
            valid_count = count_folders(directory)
            print(f"Number of files in valid: {valid_count}")

        directory = f'{root}/test/processed/'
        if os.path.exists(directory):
            test_count = count_folders(directory)
            print(f"Number of files in test: {test_count}")

        total_nb = train_count + valid_count + test_count

        print(f"Total Number of Images: {total_nb}")
        print(f"Proportions: train {train_count / total_nb:.2f} - valid {valid_count / total_nb:.2f} - test {test_count / total_nb:.2f}")


def merge_datasets_with_same_split(root : str, datasets : List[str], verbose : bool = True, splits_ : List[float] = [0.6, 0.2, 0.2]) -> None: # for complete training
    """Merge n annotation_wise_scraper dataset into one. Copy them in a new directory.
    It is used to create large scale dataset for training and evaluation.
    In this function each of the datasets has the same split between train, val and test."""
    for i, dataset_path in tqdm(enumerate(datasets), total = len(datasets), desc = 'Merging datasets', disable = not verbose):
        dataset_files = get_file_path_list(dataset_path)
        splits_counts = get_img_per_splits(dataset_files, splits = splits_)

        for file_path in tqdm(dataset_files):
            file_name = file_path.split('/')[-1]
            img, _ = file_name.split('_')

            if int(img) in splits_counts['train']:
                new_file_path = f'{root}/train/processed/{i}_{file_name}'

            elif int(img) in splits_counts['valid']:
                new_file_path = f'{root}/valid/processed/{i}_{file_name}'

            elif int(img) in splits_counts['test']:
                new_file_path = f'{root}/test/processed/{i}_{file_name}'

            else:
                ValueError("The image is not in any set !")

            copy_file(file_path, new_file_path)

    if verbose:
        train_count = 0
        valid_count = 0
        test_count = 0

        directory = f'{root}/train/processed/'
        if os.path.exists(directory):
            train_count = count_folders(directory)
            print(f"Number of files in train: {train_count}")

        directory = f'{root}/valid/processed/'
        if os.path.exists(directory):
            valid_count = count_folders(directory)
            print(f"Number of files in valid: {valid_count}")

        directory = f'{root}/test/processed/'
        if os.path.exists(directory):
            test_count = count_folders(directory)
            print(f"Number of files in test: {test_count}")

        total_nb = train_count + valid_count + test_count

        print(f"Total Number of Images: {total_nb}")
        print(f"Proportions: train {train_count / total_nb:.2f} - valid {valid_count / total_nb:.2f} - test {test_count / total_nb:.2f}")


def count_folders(directory: str) -> int:
    """
    Function to count the nb of folders in a directory.

    Args:
        directory (str): path to the directory.

    Returns:
        int: the nb of folders.
    """
    return len([f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))])


def get_file_path_list(root : str) -> List[str]:
    """Get the list of files in a directory.
    root: str, path to the directory
    Returns: List[str], list of file paths."""

    return [f'{root}/{file}' for file in os.listdir(root)]


def copy_file(file_path : str, new_file_path : str) -> None:
    """Copy a file to a new location.
    file_path: str, path to the file to copy
    new_file_path: str, path to the new location."""
    os.makedirs(new_file_path, exist_ok = True)

    shutil.copy(file_path + '/img.jpg', new_file_path + '/img.jpg')
    shutil.copy(file_path + '/mask.jpg', new_file_path + '/mask.jpg')
