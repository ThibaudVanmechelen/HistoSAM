'''Dataset class for SAM dataset to use in PyTorch. Please make sure that your images are 1024x1024 pixels to prevent any problems with the model performances.'''
import os
from abc import ABC, abstractmethod
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from cv2 import (
    INTER_NEAREST,
    MORPH_ELLIPSE,
    dilate,
    erode,
    getStructuringElement,
    line,
    resize,
)
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset
from tqdm import tqdm

from .preprocess import to_dict

IMG_RESOLUTION = 1024
INPUT_MASK_RESOLUTION = 256

class AbstractSAMDataset(Dataset, ABC):
    '''Abstract class for SAM dataset. Checkout SAMDataset for more information about implementation'''

    @abstractmethod
    def _load_data(self):
        '''Load images and masks'''
        pass

    def _load_prompt(self, zoom_out:float=1.0, n_points:int=1, n_neg_points:int=1, inside_box:bool=False, near_center:float=-1, random_box_shift:int=0, mask_prompt_type:str='truth', box_around_mask:bool=False) -> dict[str, np.ndarray]:
        '''Compute and load prompts for the dataset'''
        self.prompts = {'points':[None for _ in range(len(self.images))], 
                   'box':[None for _ in range(len(self.images))], 
                   'neg_points':[None for _ in range(len(self.images))],
                   'mask':[None for _ in range(len(self.images))]}
        if self.prompt_type['mask']:
            self.prompts['mask'] = np.array([self._get_mask(self.masks[i], i, mask_prompt_type) for i in tqdm(range(len(self.images)), desc='Computing masks...', total=len(self.images), disable=not self.verbose)])
        if self.prompt_type['box']:
            self.prompts['box'] = np.array([self._get_box(self.masks[i], i, zoom_out, random_box_shift, box_around_mask) for i in tqdm(range(len(self.images)), desc='Computing boxes...', total=len(self.images), disable=not self.verbose)])
        if self.prompt_type['neg_points']:
            self.prompts['neg_points'] = np.array([self._get_negative_points(self.masks[i], i, n_neg_points, inside_box) for i in tqdm(range(len(self.images)), desc='Computing negative points...', total=len(self.images), disable=not self.verbose)])
        if self.prompt_type['points']:
            self.prompts['points'] = np.array([self._get_points(self.masks[i], i, n_points, near_center) for i in tqdm(range(len(self.images)), desc='Computing points...', total=len(self.images), disable=not self.verbose)])

    def _get_points(self, mask_path:str, index:int, n_points:int=1, near_center:float=-1):
        '''Get n_points points from the mask'''
        mask = plt.imread(mask_path)
        idx = np.arange(mask.shape[0]*mask.shape[1])
        flatten_mask = mask.flatten()
        if near_center <= 0:
            points = np.random.choice(idx, n_points, p=flatten_mask/flatten_mask.sum())
        else:
            distance = (distance_transform_edt(mask)**near_center).flatten()
            points = np.random.choice(idx, n_points, p=(distance/distance.sum()))
        x, y = np.unravel_index(points, mask.shape)
        points = np.stack((y, x), axis=1)
        return points

    def _get_negative_points(self, mask_path:str, index:int, n_neg_points:int, inside_box:bool=False, zoom_out:float=1.0, random_box_shift:int=0, box_around_mask:bool=False, mask_prompt_type:str='truth'):
        '''Get n_neg_points points outside the mask
        n_neg_points: int, number of negative points to get
        inside_box: bool, if True, negative points will be inside the bounding box of the mask (but still outside the mask),
        Please refer to _get_box as the other parameters correspond to the same parameters in _get_box.'''
        mask = plt.imread(mask_path)
        idx = np.arange(mask.shape[0]*mask.shape[1])
        flatten_mask = mask.flatten()
        flatten_mask = np.where(flatten_mask > 0, 1, 0)
        if inside_box:
            dilation = 100 # dilation is applied to both box_mask and mask to avoid points near the true mask
            if self.prompts['box'][index] is None:
                x_min, y_min, x_max, y_max = self._get_box(mask_path, index, zoom_out, random_box_shift, box_around_mask, mask_prompt_type)
            else:
                x_min, y_min, x_max, y_max = self.prompts['box'][index]
            box_mask = np.ones_like(mask) * 255
            box_mask[y_min:y_max, x_min:x_max] = 0
            kernel_box = getStructuringElement(MORPH_ELLIPSE, (int(dilation * 1.2), int(dilation * 1.2)))
            kernel_item = getStructuringElement(MORPH_ELLIPSE, (dilation, dilation))
            flatten_mask = erode(box_mask, kernel_box).flatten() + dilate(mask, kernel_item).flatten()
            flatten_mask = np.where(flatten_mask > 0, 1, 0)
        probabilities = (1 - flatten_mask)/((1 - flatten_mask).sum())
        points = np.random.choice(idx, n_neg_points, p=probabilities)
        x, y = np.unravel_index(points, mask.shape)
        points = np.stack((y, x), axis=1)
        return points
    
    def _get_box(self, mask_path:str, index:int, zoom_out:float=1.0, random_box_shift:int=0, box_around_mask:bool=False, mask_prompt_type:str='truth') -> tuple[int, int, int, int]:
        '''Get a box from the mask
        zoom_out: float, factor to zoom out the box. Add a margin around the mask.
        random_box_shift: int, if greater than 0, the bounding box corners will be shifted randomly by a value between -random_box_shift and random_box_shift
        box_around_mask: bool, if True, the box will be around the input mask.'''
        assert zoom_out > 0, 'Zoom out factor must be greater than 0'
        if box_around_mask:
            if self.prompts['mask'][index] is None:
                mask = self._get_mask(mask_path, index, mask_prompt_type=mask_prompt_type)
            else:
                mask = self.prompts['mask'][index]
            mask = resize(mask, (IMG_RESOLUTION, IMG_RESOLUTION), interpolation=INTER_NEAREST)
        else:
            mask = plt.imread(mask_path)
        x_min, y_min, x_max, y_max = mask.shape[0], mask.shape[1], 0, 0
        h_sum = mask.sum(axis=1)
        v_sum = mask.sum(axis=0)
        x_min = np.argmax(v_sum > 0)
        x_max = len(v_sum) - np.argmax(v_sum[::-1] > 0)
        y_min = np.argmax(h_sum > 0)
        y_max = len(h_sum) - np.argmax(h_sum[::-1] > 0)
        box_width = x_max - x_min
        box_height = y_max - y_min
        h_padding = (box_width * zoom_out - box_width)  / 2
        v_padding = (box_height * zoom_out - box_height) / 2
        shifts = [0, 0, 0, 0]
        if random_box_shift > 0:
            shifts = np.random.randint(-random_box_shift, 2*random_box_shift, 4)
        x_min = max(0, x_min - h_padding + shifts[0])
        x_max = min(mask.shape[0], x_max + h_padding + shifts[1])
        y_min = max(0, y_min - v_padding + shifts[2])
        y_max = min(mask.shape[1], y_max + v_padding + shifts[3])
        return int(x_min), int(y_min), int(x_max), int(y_max)

    def _get_mask(self, mask_path:str, index:int, mask_prompt_type:str='truth') -> np.ndarray:
        '''Get the mask from the mask path/
        mask_prompt_type: str, type of mask to use for automatic annotation. Can be 'truth' or 'morphology' or 'scribble'. Default is 'truth'.
        'truth': the mask is the truth mask, 'morphology': the mask is the truth mask dilated by a 40x40 kernel, 'scribble': the mask is a simulated scribbled mask by the user.'''
        mask = resize(plt.imread(mask_path), (INPUT_MASK_RESOLUTION, INPUT_MASK_RESOLUTION), interpolation=INTER_NEAREST)
        if mask_prompt_type == 'truth':
            return np.where(mask > 0, 1, 0)
        if mask_prompt_type == 'morphology':
            return self._get_mask_morphology(mask)
        if mask_prompt_type == 'scribble':
            return self._get_mask_scribble(mask)
        raise ValueError('mask_prompt_type must be truth, morphology or scribble')
    
    def _get_mask_morphology(self, mask:np.ndarray) -> np.ndarray:
        n = 40
        kernel = getStructuringElement(MORPH_ELLIPSE, (n, n))
        return dilate(mask, kernel=kernel)
    
    def _get_mask_scribble(self, mask:np.ndarray) -> np.ndarray:
        '''Get the scribble mask from the mask. n points are chosen on each side (left and right) of the mask. Then
        a line is drawn between corresponding left and right points to simulate human drawn scribbles.
        mask: np.ndarray, the mask to scribble on.
        '''
        n = 8
        h_sum = mask.sum(axis=1)
        y_min = np.argmax(h_sum > 0)
        y_max = len(h_sum) - np.argmax(h_sum[::-1] > 0)
        height = y_max - y_min
        left_points_y = np.linspace(y_min +0.05 * height, y_max - 0.1, n, dtype=int)
        right_points_y = np.linspace(y_min + 0.1, y_max-0.05, n, dtype=int)
        left_points_x = np.array([np.argmax(mask[y] > 0) for y in left_points_y])
        right_points_x = np.array([len(mask[y]) - np.argmax(mask[y][::-1] > 0) for y in right_points_y])
        # Draw horizontal lines between the left and right points
        scribble_mask = np.zeros_like(mask)
        for i in range(1, n-1):
            line(scribble_mask, (left_points_x[i], left_points_y[i]), (right_points_x[i], right_points_y[i]), 1, 1)
        kernel = getStructuringElement(MORPH_ELLIPSE, (20, 20))
        return dilate(scribble_mask, kernel=kernel)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx:int) -> tuple:
        img = plt.imread(self.images[idx])
        mask = plt.imread(self.masks[idx])
        prompt = {'points':self.prompts['points'][idx], 'box':self.prompts['box'][idx], 'neg_points':self.prompts['neg_points'][idx], 'mask':self.prompts['mask'][idx]}
        return img, np.where(mask > 0, 1, 0), prompt

class SAMDataset(AbstractSAMDataset):
    '''Prepare a dataset for segmentation by Segment Anything Model. Checkout AbstractSAMDataset for more information'''

    def __init__(self, root:str, transform=None, use_img_embeddings:bool=False,
                 prompt_type:dict={'points':False, 'box': False, 'neg_points':False, 'mask':False}, 
                 n_points:int=1, n_neg_points:int=1, zoom_out:float=1.0, verbose:bool=False, 
                 random_state:int=None, to_dict:bool=True, neg_points_inside_box:bool=False, 
                 points_near_center:float=-1, random_box_shift:int=0, mask_prompt_type:str='truth', 
                 box_around_mask:bool=False):
        '''Initialize SAMDataset class.
        root: str, path to the dataset directory
        transform: callable, transform to apply to the images and masks
        use_img_embeddings: bool, if True, the model will use image embeddings instead of the images
        prompt_type: Dict[str, bool], type of automatic annotation to use
        n_points: int, number of points to use for automatic annotation if prompt is 'points'.
        n_neg_points: int, number of negative points to use for automatic annotation if prompt is 'neg_points'.
        zoom_out: float, factor to zoom out the bounding box. Add a margin around the mask.
        verbose: bool, if True, print progress messages
        random_state: int, random state for reproducibility
        to_dict: bool, if True, the __getitem__ method will return a dictionary with the image, the prompt and the mask. If False, it will return the image, the mask and the prompt.
        neg_points_inside_box: bool, if True, negative points will be inside the bounding box of the mask
        points_near_center: float, if greater than 0, points will be more likely to be near the center of the mask
        random_box_shift: int, if greater than 0, the bounding box corners will be shifted randomly by a value between -random_box_shift and random_box_shift
        mask_prompt_type: str, type of mask to use for automatic annotation. Can be 'truth' or 'morphology' or 'scribble'. Default is 'truth'.
        box_around_mask: bool, if True, the box will be around the input mask.
        '''
        self.root = root
        self.transform = transform
        self.use_img_embeddings = use_img_embeddings
        self.images = []
        self.masks = []
        self.prompts = []
        self.prompt_type = prompt_type
        self.verbose = verbose
        self.n_points = n_points
        self.n_neg_points = n_neg_points
        self.near_center = points_near_center
        self.inside_box = neg_points_inside_box
        self.to_dict = to_dict
        self.zoom_out = zoom_out
        if random_state is not None:
            torch.manual_seed(random_state)
        if self.verbose:
            print('Loading images and masks paths...')
        self._load_data()
        self._load_prompt(zoom_out=zoom_out, n_points=n_points, n_neg_points=n_neg_points, inside_box=neg_points_inside_box, near_center=points_near_center, random_box_shift=random_box_shift, mask_prompt_type=mask_prompt_type, box_around_mask=box_around_mask)
        if self.use_img_embeddings:
            self._load_img_embeddings()
        if self.verbose:
            print('Done!')

    def _load_data(self):
        '''Load images and masks'''
        for f in os.listdir(self.root + 'processed/'):
            for g in os.listdir(self.root + 'processed/' + f):
                if g.endswith('.jpg'):
                    if 'mask' in g:
                        self.masks.append(self.root + 'processed/' + f + '/' + g)
                    else:
                        self.images.append(self.root + 'processed/' + f + '/' + g)

    def _load_img_embeddings(self):
        '''Load image embeddings'''
        self.img_embeddings = []
        for f in os.listdir(self.root + 'img_embeddings/'):
            print('Not working')
            self.img_embeddings.append(torch.load(self.root + 'img_embeddings/' + f).to('cpu'))

    def __getitem__(self, idx:int) -> tuple:
        img = plt.imread(self.images[idx])
        mask = plt.imread(self.masks[idx])
        prompt = {'points':self.prompts['points'][idx], 'box':self.prompts['box'][idx], 'neg_points':self.prompts['neg_points'][idx], 'mask':self.prompts['mask'][idx]}
        if self.use_img_embeddings:
            return to_dict(self.img_embeddings[idx], prompt, self.use_img_embeddings), np.where(mask > 0, 1, 0)
        if self.transform:
            img, mask = self.transform(img, mask)
        if self.to_dict:
            return to_dict(img, prompt), np.where(mask > 0, 1, 0)
        return img, np.where(mask > 0, 1, 0), prompt
    

class AugmentedSamDataset(SAMDataset):
    '''Prepare an augmented dataset for segmentation by Segment Anything Model. 
    Checkout AbstractSAMDataset for more information.
    Each samples exist in multiple versions with different prompts.'''

    def __init__(self, root:str, use_img_embeddings:bool=True,
                 n_points:int=1, n_neg_points:int=1, zoom_out:float=1.0, verbose:bool=False, 
                 random_state:int=None, to_dict:bool=True, random_box_shift:int=20, mask_prompt_type:str='truth',
                 load_on_cpu:bool=False, filter_files:callable=None):
        '''Initialize SAMDataset class.
        root: str, path to the dataset directory
        transform: callable, transform to apply to the images and masks
        use_img_embeddings: bool, if True, the model will use image embeddings instead of the images
        n_points: int, number of points to use for automatic annotation if prompt is 'points'.
        n_neg_points: int, number of negative points to use for automatic annotation if prompt is 'neg_points'.
        zoom_out: float, factor to zoom out the bounding box. Add a margin around the mask.
        verbose: bool, if True, print progress messages
        random_state: int, random state for reproducibility
        to_dict: bool, if True, the __getitem__ method will return a dictionary with the image, the prompt and the mask. If False, it will return the image, the mask and the prompt.
        random_box_shift: int, if greater than 0, the bounding box corners will be shifted randomly by a value between -random_box_shift and random_box_shift
        mask_prompt_type: str, type of mask to use for automatic annotation. Can be 'truth' or 'morphology' or 'scribble'. Default is 'truth'.
        load_on_cpu: bool, if True, the entire dataset will be loaded on the CPU RAM. Data loading will be faster but will consume more memory. Default: False
        '''
        prompt_type = {'points':False, 'box':False, 'neg_points':False, 'mask':False}
        self.filter_files = filter_files
        neg_points_inside_box = True
        points_near_center = 4
        box_around_mask = False
        super().__init__(root, use_img_embeddings=use_img_embeddings, 
                         prompt_type=prompt_type, n_points=n_points, 
                         n_neg_points=n_neg_points, zoom_out=zoom_out, 
                         verbose=verbose, random_state=random_state, 
                         to_dict=to_dict, neg_points_inside_box=neg_points_inside_box, 
                         points_near_center=points_near_center, 
                         random_box_shift=random_box_shift, 
                         mask_prompt_type=mask_prompt_type, 
                         box_around_mask=box_around_mask)
        self.load_on_cpu = load_on_cpu
        if load_on_cpu:
            self.images = [plt.imread(img) for img in self.images]
            self.masks = [plt.imread(mask) for mask in self.masks]
        self.prompts = torch.load(self.root + 'prompts.pt')
        prompt_type = {'points':True, 'box':True, 'neg_points':True, 'mask':True}
        

    def __getitem__(self, idx:int) -> tuple:
        img_idx = idx % len(self.images)
        prompt_idx = idx // len(self.images)
        if self.load_on_cpu:
            img = self.images[img_idx]
            mask = self.masks[img_idx]
        else:
            img = plt.imread(self.images[img_idx])
            mask = plt.imread(self.masks[img_idx])
        prompt = {'points':None, 'box':None, 'neg_points':None, 'mask':None}
        prompts_combinaisons = [['mask', 'points', 'neg_points'],
                                ['mask'],
                                ['box'],
                                ['box', 'points', 'neg_points']]
        for key in prompts_combinaisons[prompt_idx]:
            prompt[key] = self.prompts[key][img_idx]
        if self.use_img_embeddings:
            return to_dict(self.img_embeddings[img_idx], prompt, self.use_img_embeddings), np.where(mask > 0, 1, 0)
        if self.transform:
            img, mask = self.transform(img, mask)
        if self.to_dict:
            return to_dict(img, prompt), np.where(mask > 0, 1, 0)
        return img, np.where(mask > 0, 1, 0), prompt
    
    def _load_data(self):
        '''Load images and masks. Allows to filter files.'''
    
        for f in os.listdir(self.root + 'processed/'):
            print(f)
            if self.filter_files is not None:
                if not self.filter_files(f):
                    continue
            for g in os.listdir(self.root + 'processed/' + f):
                if g.endswith('.jpg'):
                    if 'mask' in g:
                        self.masks.append(self.root + 'processed/' + f + '/' + g)
                    else:
                        self.images.append(self.root + 'processed/' + f + '/' + g)
    
    def _load_img_embeddings(self):
        '''Load image embeddings'''
        self.img_embeddings = []
        for f in os.listdir(self.root + 'img_embeddings/'):
            if self.filter_files is not None:
                if not self.filter_files(f):
                    continue
            self.img_embeddings.append(torch.load(self.root + 'img_embeddings/' + f).to('cpu'))

    def __len__(self):
        return len(self.images) * 4

class SamDatasetFromFiles(AbstractSAMDataset):
    '''Prepare a dataset for segmentation by Segment Anything Model. Checkout AbstractSAMDataset for more information'''

    def __init__(self, root:str, transform=None, use_img_embeddings:bool=False,
                 prompt_type:dict={'points':False, 'box': False, 'neg_points':False, 'mask':False}, 
                 n_points:int=1, n_neg_points:int=1, zoom_out:float=1.0, verbose:bool=False, 
                 random_state:int=None, to_dict:bool=True, neg_points_inside_box:bool=False, 
                 points_near_center:float=-1, random_box_shift:int=0, mask_prompt_type:str='truth', 
                 box_around_mask:bool=False, filter_files:callable=None, load_on_cpu:bool=False):
        '''Initialize SAMDataset class.
        Can only be initialized from files obtained from save_img_embeddings.py script.
        '''
        self.root = root
        self.transform = transform
        self.use_img_embeddings = use_img_embeddings
        self.images = []
        self.masks = []
        self.prompts = []
        self.prompt_type = prompt_type
        self.verbose = verbose
        self.n_points = n_points
        self.n_neg_points = n_neg_points
        self.near_center = points_near_center
        self.inside_box = neg_points_inside_box
        self.to_dict = to_dict
        self.zoom_out = zoom_out
        self.prompts = torch.load(self.root + 'prompts.pt')
        print(len(self.prompts))
        prompt_type = {'points':True, 'box':True, 'neg_points':True, 'mask':True}
        self.load_on_cpu = load_on_cpu
        self.filter_files = filter_files
        if random_state is not None:
            torch.manual_seed(random_state)
        if self.verbose:
            print('Loading images and masks paths...')
        self._load_data()
        if self.use_img_embeddings:
            self._load_img_embeddings()
        if load_on_cpu:
            self.images = [plt.imread(img) for img in self.images]
            self.masks = [plt.imread(mask) for mask in self.masks]
        if self.verbose:
            print('Done!')

    def _load_data(self):
        '''Load images and masks. Allows to filter files.'''
        prompts_to_delete = []
        for i, f in enumerate(os.listdir(self.root + 'processed/')):
            
            if self.filter_files is not None:
                if not self.filter_files(f):
                    prompts_to_delete.append(i)
                    continue
            for g in os.listdir(self.root + 'processed/' + f):
                if g.endswith('.jpg'):
                    if 'mask' in g:
                        self.masks.append(self.root + 'processed/' + f + '/' + g)
                    else:
                        self.images.append(self.root + 'processed/' + f + '/' + g)
        for key in self.prompts:
            self.prompts[key] = np.delete(self.prompts[key], prompts_to_delete, axis=0)
    
    def _load_img_embeddings(self):
        '''Load image embeddings'''
        self.img_embeddings = []
        
        for i, f in enumerate(os.listdir(self.root + 'img_embeddings/')):
            if self.filter_files is not None: 
                if not self.filter_files(f):
                    continue
            self.img_embeddings.append(torch.load(self.root + 'img_embeddings/' + f).to('cpu'))
        

    def __getitem__(self, idx:int) -> tuple:
        img_idx = idx % len(self.images)
        prompt_idx = idx // len(self.images)
        if self.load_on_cpu:
            img = self.images[img_idx]
            mask = self.masks[img_idx]
        else:
            img = plt.imread(self.images[img_idx])
            mask = plt.imread(self.masks[img_idx])
        prompt = {'points':None, 'box':None, 'neg_points':None, 'mask':None}
        prompts_combinaisons = [['mask', 'points', 'neg_points'],
                                ['mask'],
                                ['box'],
                                ['box', 'points', 'neg_points']]
        for key in prompts_combinaisons[prompt_idx]:
            prompt[key] = self.prompts[key][img_idx]
        if self.use_img_embeddings:
            return to_dict(self.img_embeddings[img_idx], prompt, self.use_img_embeddings), np.where(mask > 0, 1, 0)
        if self.transform:
            img, mask = self.transform(img, mask)
        if self.to_dict:
            return to_dict(img, prompt), np.where(mask > 0, 1, 0)
        return img, np.where(mask > 0, 1, 0), prompt
    
    def __len__(self):
            return len(self.images) * 4

def filter_dataset(file_name, datasets:list):
    '''Filter datasets to keep only the ones.
    datasets: List[str], list of boolean values to filter, True if the dataset is to keep, False otherwise.'''
    file_name_dataset = file_name.split('_')[0]
    return datasets[int(file_name_dataset)]

