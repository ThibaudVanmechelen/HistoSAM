import os
import shutil

import random
from random import choice, shuffle
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
                                         datasets_splits : List[List[float]], verbose : bool = True) -> None: # for generalisation TODO update documentation
    # Note validation set should come from the same datasets use for the training set in order to respoect data distribution, it must not come from the test set.
    """Merge n annotation_wise_scraper dataset into one. Copy them in a new directory.
    It is used to create large scale dataset for training and evaluation.
    root: str, path to the new directory where the datasets will be copied.
    datasets: List[str], list of paths to the datasets to merge."""
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
        directory = f'{root}/train/processed/'
        train_count = count_folders(directory)
        print(f"Number of files in train: {train_count}")

        directory = f'{root}/valid/processed/'
        valid_count = count_folders(directory)
        print(f"Number of files in valid: {valid_count}")

        directory = f'{root}/test/processed/'
        test_count = count_folders(directory)
        print(f"Number of files in test: {test_count}")

        total_nb = train_count + valid_count + test_count

        print(f"Total Number of Images: {total_nb}")
        print(f"Proportions: train {train_count / total_nb:.2f} - valid {valid_count / total_nb:.2f} - test {test_count / total_nb:.2f}")


def merge_datasets_with_same_split(root : str, datasets : List[str], verbose : bool = True, splits = [0.6, 0.2, 0.2]) -> None: # for complete training
    for i, dataset_path in tqdm(enumerate(datasets), total = len(datasets), desc = 'Merging datasets', disable = not verbose):
        dataset_files = get_file_path_list(dataset_path)
        splits_counts = get_img_per_splits(dataset_files, splits = splits)

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
        directory = f'{root}/train/processed/'
        train_count = count_folders(directory)
        print(f"Number of files in train: {train_count}")

        directory = f'{root}/valid/processed/'
        valid_count = count_folders(directory)
        print(f"Number of files in valid: {valid_count}")

        directory = f'{root}/test/processed/'
        test_count = count_folders(directory)
        print(f"Number of files in test: {test_count}")

        total_nb = train_count + valid_count + test_count

        print(f"Total Number of Images: {total_nb}")
        print(f"Proportions: train {train_count / total_nb:.2f} - valid {valid_count / total_nb:.2f} - test {test_count / total_nb:.2f}")


def count_folders(directory: str) -> int:
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


# def split_dataset_into_folds(dataset_files : List[str], k : int = 5, seed : int = 42) -> List[List[str]]:
#     img_per_slide_count = get_img_per_slide_count(dataset_files)
#     distinct_img = list(img_per_slide_count.keys())

#     random.seed(seed)
#     shuffle(distinct_img)

#     folds = [[] for _ in range(k)]
#     fold_annotation_counts = [0] * k

#     for img in distinct_img:
#         min_fold = fold_annotation_counts.index(min(fold_annotation_counts))

#         folds[min_fold].append(img)
#         fold_annotation_counts[min_fold] += img_per_slide_count[img] # try to have fold with same number of annotations

#     return folds


# def get_files_for_fold(dataset_files : List[str], fold : List[str]) -> List[str]:
#     files_for_fold = []

#     for file_path in dataset_files:
#         file_name = file_path.split('/')[-1]
#         img, _ = file_name.split('_')

#         if img in fold:
#             files_for_fold.append(file_path)

#     return files_for_fold


# def cross_validate_datasets(root : str, datasets : List[str], k : int = 5, verbose : bool = True):
#     dataset_splits = {name : split_dataset_into_folds(get_file_path_list(name), k) for name in datasets}
    
#     for i in range(k):
#         if verbose:
#             print(f"Processing fold {i + 1}/{k}")
        
#         train_files, valid_files, test_files = [], [], []

#         for dataset_name, folds in dataset_splits.items():
#             dataset_files = get_file_path_list(dataset_name)
            
#             test_files.extend(get_files_for_fold(dataset_files, folds[i]))
#             valid_files.extend(get_files_for_fold(dataset_files, folds[(i + 1) % k]))
            
#             for j, fold in enumerate(folds):
#                 if j != i and j != (i + 1) % k:
#                     train_files.extend(get_files_for_fold(dataset_files, fold))

#         save_fold_to_directory(root, i, train_files, "train")
#         save_fold_to_directory(root, i, valid_files, "valid")
#         save_fold_to_directory(root, i, test_files, "test")


# def save_fold_to_directory(root : str, fold_index : int, files : List[str], split : str):
#     for file_path in files:
#         file_name = file_path.split('/')[-1]
#         new_file_path = os.path.join(root, f"fold_{fold_index}/{split}/", file_name)

#         os.makedirs(os.path.dirname(new_file_path), exist_ok = True)
#         copy_file(file_path, new_file_path)


if __name__ == '__main__':
    root = '../../datasets/all/'
    datasets = ['../../datasets/camelyon16.1/processed/', '../../datasets/LBTD-NEO04/processed/', '../../datasets/LBTD-AGDC10/processed/']
    #datasets = ['../../datasets/void/processed/', '../../datasets/void/processed/', '../../datasets/LBTD-AGDC10/processed/']
    merge_datasets(root, datasets)