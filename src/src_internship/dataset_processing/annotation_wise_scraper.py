'''Download cropped images around annotations from a Cytomine project.
This script is made to be used on any Cytomine project to download images and masks around annotations.
Crops around annotations are resized into 1024x1024 images.'''

import logging
import os
import sys
from argparse import ArgumentParser
from math import floor
from tomllib import load

import matplotlib.pyplot as plt
import numpy as np
from box import Box
from cytomine import Cytomine
from cytomine.models import (
    Annotation,
    AnnotationCollection,
    ImageInstance,
    ImageInstanceCollection,
)
from shapely import wkt
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm


def load_config(config_path:str):
    '''Loads a config file in toml format. Returns a dictionary with the config values.'''
    with open(config_path, "rb") as f:
        config = load(f)
    return Box(config)


def get_localisation_of_annotation(annotation:Annotation) -> tuple[int, int, int, int]:
    '''Returns the width, height and top left corner of an annotation.
    annotation: Annotation, the annotation to get the dimensions from.
    Returns: width, height, top-left x, top-left y.'''
    geometry = wkt.loads(annotation.location).bounds
    width = floor(geometry[2] - geometry[0])
    height = floor(geometry[3] - geometry[1])
    return width, height, floor(geometry[0]), floor(geometry[3])

def get_annotation_size(img_width:int, img_height:int, width:int, height:int, zoom_out_factor:float=1.0, minimum_size:int=-1) -> int:
    '''Returns the size (side length) of an annotation.
    img_width: int, width of the image
    img_height: int, height of the image
    width: int, width of the annotation
    height: int, height of the annotation
    zoom_out_factor: float, factor to zoom out the annotation
    minimum_size: int, minimum size of the annotation (in case width and height are too small). Negative values means no minimum size.
    '''
    annotation_size = max(width, height)
    zoomed_size = max(annotation_size * zoom_out_factor, minimum_size)
    return min(floor(zoomed_size), img_width, img_height)

def get_random_shift(random_shift:int, random_state:int) -> np.ndarray:
    '''Returns a random shift for the ROI.
    random_shift: int, maximum shift value
    random_state: int, random state for reproducibility of the shift.
    Returns: random shift as a numpy array.'''
    np.random.seed(random_state)
    if random_shift > 0:
        shift = np.random.randint(0, (np.abs(random_shift)) * 2, size=(2,)) - random_shift
    else:
        shift = np.array([0, 0])
    return shift
    
def get_roi_around_annotation(img:ImageInstance, annotation:Annotation, config:dict) -> tuple[int, int, int, int]:
    '''Returns a valid region of interest (ROI) of a gigapixel image around an annotation.
    img: ImageInstance, the image to crop
    annotation: Annotation, the annotation to crop around
    config: dict, configuration dictionary
    random_shift: int, random shift to select the ROI
    random_state: int, random state for reproducibility of random shift.
    Returns: x, y, width, height of the ROI'''
    random_shift = config['cytomine']['random_shift']
    random_state = config['cytomine']['random_state']
    zoom_out_factor = config['cytomine']['zoom_out_factor']
    minimum_size = -1
    assert zoom_out_factor >= 1, 'Zoom out factor must be greater or equal to one'
    assert random_shift < img.width and random_shift < img.height, 'Random shift must be smaller than the image size'
    # Get location of the annotation
    annotation_width, annotation_height, x, y = get_localisation_of_annotation(annotation)
    size = get_annotation_size(img.width, img.height, annotation_width, annotation_height, zoom_out_factor, minimum_size)
    shift = get_random_shift(random_shift, random_state)
    h = (size - annotation_width) / 2 
    v = (size - annotation_height) / 2
    x = floor(x - h + shift[0])
    y = floor(img.height - y - v + shift[1])
    # Correct the ROI if it is out of the image
    x = min(x, img.width - size)
    y = min(y, img.height - size)
    x = max(0, x)
    y = max(0, y)
    return x, y, size, size

def is_mask_empty(mask_array:np.ndarray) -> bool:
    '''Check if a mask is empty (all zeros).
    mask_array: np.ndarray, the mask to check.
    Returns: bool, True if the mask is empty, False otherwise.'''
    return len(mask_array.shape) == 3 or mask_array.sum() == 0

def is_mask_resolution_valid(mask_array:np.ndarray, input_size:int) -> bool:
    '''Check if the resolution of the mask is valid.
    mask_array: np.ndarray, the mask to check
    input_size: int, size of the input of the model
    Returns: bool, True if the resolution is valid, False otherwise.'''
    return mask_array.shape[0] == input_size and mask_array.shape[1] == input_size

def delete_sample(dataset_path:str, i:int, a:int):
    '''Delete a sample from the dataset.
    dataset_path: str, path to the dataset
    i: int, index of the image
    a: int, index of the annotation'''
    os.remove(dataset_path + f'{i}_{a}/mask.jpg')
    os.remove(dataset_path + f'{i}_{a}/img.jpg')
    os.rmdir(dataset_path + f'{i}_{a}')

def download_images(config:dict):
    '''Downloads all images from a Cytomine projects around annotations.
    Requires a config dictionary (see config.toml and load_config function).
    Must be executed unside a with Cytomine() statement.'''
    img_collections = ImageInstanceCollection().fetch_with_filter("project", config['cytomine']['project_id'])
    dataset_path = '../' + config.cytomine.dataset_path + 'processed/'
    input_size = config.sam.input_size
    print('Croping and downloading images')
    for i,img in tqdm(enumerate(img_collections), total=len(img_collections)):
        annotations = AnnotationCollection()
        annotations.project = config.cytomine.project_id
        annotations.image = img.id
        annotations.users = config.cytomine.annotation_users_id
        annotations.showWKT = True
        annotations.showMeta = True
        annotations.showGIS = True
        annotations.fetch()
        for a, annotation in enumerate(annotations):
            x, y, width, height = get_roi_around_annotation(img, annotation, config)
            img.window(x, y, width, height, dest_pattern=dataset_path + f'{i}_{a}/img', annotations=[annotation.id], max_size=1024)
            img.window(x, y, width, height, mask=True, dest_pattern=dataset_path + f'{i}_{a}/mask', annotations=[annotation.id], max_size=1024)
            if len(os.listdir(dataset_path + f'{i}_{a}')) == 0:
                os.rmdir(dataset_path + f'{i}_{a}')
                print(f'Error downloading image {img.originalFilename} ({img.id}), annotation {a}')
                continue
            mask_array = plt.imread(dataset_path + f'{i}_{a}/mask.jpg')
            if is_mask_empty(mask_array):
                delete_sample(dataset_path, i, a)
            # SAM input size is 1024x1024
            else:
                if not is_mask_resolution_valid(mask_array, input_size):
                    img_array = plt.imread(dataset_path + f'{i}_{a}/img.jpg')
                    print(f'Resizing image {img.originalFilename} ({img.id}), annotation {a}, from {img_array.shape} to {input_size}x{input_size}')
                    resized_img = F.resize(F.to_pil_image(img_array), (input_size, input_size))
                    resized_img.save(dataset_path + f'{i}_{a}/img.jpg')
                    resized_mask = F.resize(F.to_pil_image(mask_array), (input_size, input_size), interpolation=InterpolationMode.NEAREST)
                    resized_mask.save(dataset_path + f'{i}_{a}/mask.jpg')


if __name__ == '__main__':
    parser = ArgumentParser(description='Download cropped images around annotations from a Cytomine project.')
    parser.add_argument('--config', required=False, type=str, help='Path to the configuration file. Default: ../config.toml', default='../config.toml')
    args = parser.parse_args()
    keys = load_config('../keys.toml')
    config = load_config(args.config)
    with Cytomine(keys['host'], keys['public_key'], keys['private_key'], verbose=logging.ERROR) as cytomine:
        download_images(config)
