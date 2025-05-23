"""Downloads ROI from images from a specific Cytomine project"""

import os
import shutil
from math import floor
from queue import Queue

import matplotlib.pyplot as plt
from cytomine.models import (
    AnnotationCollection,
    ImageInstance,
    ImageInstanceCollection,
    TermCollection,
)
from shapely import wkt
from tqdm import tqdm


def get_roi(img : ImageInstance, config : dict) -> AnnotationCollection:
    """Returns the region of interest (ROI) of an image."""

    roi = AnnotationCollection()

    roi.project = config['cytomine']['project_id']
    roi.user = config['cytomine']['annotation_user_id']

    roi.image = img.id
    roi.term = config['cytomine']['ROI_term_id']

    roi.showWKT = True
    roi.showMeta = True
    roi.showGIS = True

    roi.fetch()

    return roi


def download_images(config : dict):
    """Downloads all images from a Cytomine project.
    Requires a config dictionary (see config.toml and load_config function).
    Must be executed unside a with Cytomine() statement.

    Structure of the created directory:
    raw/
    ├── img0/
    │   ├── roi0/
    │   │   ├── img.jpg
    │   │   ├── mask0.jpg
    │   │   ├── mask1.jpg
    │   │   └── ... 
    │   └── roi1/
    │       └── ...
    └── img1/
        └── roi0/
            └── ...
    """

    img_collections = ImageInstanceCollection().fetch_with_filter("project", config['cytomine']['project_id'])
    all_terms = TermCollection().fetch_with_filter("project", config['cytomine']['project_id'])

    valid_terms = [term.id for term in all_terms if term.id != int(config['cytomine']['ROI_term_id'])] # all other terms that are not ROI term id, 
    # we do this because ROI term id is the whole region, no purpose in selecting everything, we thus look at the other term

    dataset_path = '../' + config['cytomine']['dataset_path'] + 'raw/'

    nb_img = len(img_collections)
    print("Cropping and downloading images")
    print(f"Number of Images: {nb_img}")

    for i, img in tqdm(enumerate(img_collections), total = nb_img):
        roi_annotations = get_roi(img, config)

        for r, roi in enumerate(roi_annotations):
            geometry = wkt.loads(roi.location).bounds

            # Getting top left coordinates: (In Cytomine, geometries are referenced using a cartesian coordinate system ! https://github.com/cytomine/Cytomine-python-client/blob/master/examples/get_annotations.py)
            x = floor(geometry[0])
            y = floor(img.height - geometry[3])
    
            width = floor(geometry[2] - geometry[0])
            height = floor(geometry[3] - geometry[1])

            img.window(x, y, width, height, 
                       dest_pattern = dataset_path + f'img{i}/roi{r}/img', 
                       users = [config['cytomine']['annotation_user_id']], 
                       terms = valid_terms) #saving the img with ROI

            for t, term in enumerate(valid_terms):
                img.window(x, y, width, height, 
                           mask = True, 
                           dest_pattern = dataset_path + f'img{i}/roi{r}/mask{t}', 
                           users = [config['cytomine']['annotation_user_id']], 
                           terms = [term]) # saving each mask

                img_array = plt.imread(dataset_path + f'img{i}/roi{r}/mask{t}.jpg')

                if len(img_array.shape) == 3: # If the image is RGB then remove it
                    os.remove(dataset_path + f'img{i}/roi{r}/mask{t}.jpg')

            if len(os.listdir(dataset_path + f'img{i}/roi{r}')) == 1:
                os.remove(dataset_path + f'img{i}/roi{r}/img.jpg')
                os.rmdir(dataset_path + f'img{i}/roi{r}')


def process_dataset(config : dict):
    """Processes the dataset to create an organized dataset for machine learning.

    Structure of the created directory:
    processed/
    ├── 0/
    │   ├── img.jpg
    │   └── mask0.jpg
    ├── 1/
    │   ├── img.jpg
    │   └── mask1.jpg 
    └── ...
    """
    raw_path = '../' + config['cytomine']['dataset_path'] + 'raw/'
    processed_path = '../' + config['cytomine']['dataset_path'] + 'processed/'

    stack = Queue()
    n = 0

    images = os.listdir(raw_path)

    print("Searching for images")
    for img in tqdm(images):
        rois = os.listdir(raw_path + img)

        for roi in rois:
            masks = os.listdir(raw_path + img + '/' + roi)[1:] # [1:] to not take original img

            for mask in masks:
                img_src = raw_path + img + '/' + roi + '/' + 'img.jpg'
                img_dest = processed_path + str(n) + '/' + 'img.jpg'

                mask_src = raw_path + img + '/' + roi + '/' + mask
                mask_dest = processed_path + str(n) + '/' + mask # we create a dir with initial image and just one mask

                n += 1
    
                stack.put((img_src, img_dest, mask_src, mask_dest))

    print("Processing images")
    while not stack.empty():
        img_src, img_dest, mask_src, mask_dest = stack.get()

        os.makedirs(os.path.dirname(img_dest), exist_ok = True)

        shutil.copyfile(img_src, img_dest)
        shutil.copy(mask_src, mask_dest)

    print("Done")