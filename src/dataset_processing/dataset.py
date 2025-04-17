"""Dataset class for SAM dataset to use in PyTorch. Please make sure that your images are 1024x1024 pixels to prevent any problems with the model performances."""
import os
from abc import ABC, abstractmethod

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
    findContours,
    RETR_EXTERNAL, 
    CHAIN_APPROX_SIMPLE,
    drawContours,
    FILLED,
    convexHull
)
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset
from tqdm import tqdm
from copy import deepcopy

from .preprocess import to_dict

IMG_RESOLUTION = 1024
INPUT_MASK_RESOLUTION = 256

class AbstractSAMDataset(Dataset, ABC):
    """Abstract class for SAM dataset. Checkout SAMDataset for more information about implementation"""

    @abstractmethod
    def _load_data(self):
        """Load images and masks"""
        pass

    def _load_prompt(self, zoom_out : float = 1.0, 
                     n_points : int = 1, 
                     n_neg_points : int = 1, 
                     inside_box : bool = False, 
                     near_center : float = -1, 
                     random_box_shift : int = 0, 
                     mask_prompt_type : str = 'truth', 
                     box_around_mask : bool = False) -> dict[str, np.ndarray]:
        """Compute and load prompts for the dataset"""
        self.prompts = {'points': [None for _ in range(len(self.images))], 
                   'box': [None for _ in range(len(self.images))], 
                   'neg_points': [None for _ in range(len(self.images))],
                   'mask': [None for _ in range(len(self.images))]}
        
        if self.prompt_type['mask']:
            self.prompts['mask'] = np.array([self._get_mask(self.masks[i], i, mask_prompt_type) for i in tqdm(range(len(self.images)), desc='Computing masks...', total=len(self.images), disable=not self.verbose)])
        
        if self.prompt_type['box']:
            self.prompts['box'] = np.array([self._get_box(self.masks[i], i, zoom_out, random_box_shift, box_around_mask) for i in tqdm(range(len(self.images)), desc='Computing boxes...', total=len(self.images), disable=not self.verbose)])
        
        if self.prompt_type['neg_points']:
            self.prompts['neg_points'] = np.array([self._get_negative_points(self.masks[i], i, n_neg_points, inside_box) for i in tqdm(range(len(self.images)), desc='Computing negative points...', total=len(self.images), disable=not self.verbose)])
        
        if self.prompt_type['points']:
            self.prompts['points'] = np.array([self._get_points(self.masks[i], i, n_points, near_center) for i in tqdm(range(len(self.images)), desc='Computing points...', total=len(self.images), disable=not self.verbose)])

    def _get_points(self, mask_path : str, index : int, n_points : int = 1, near_center : float = -1, opened_mask : np.ndarray = None):
        """Get n_points points from the mask"""
        if mask_path:
            mask = plt.imread(mask_path)
        else:
            mask = opened_mask

        idx = np.arange(mask.shape[0] * mask.shape[1])
        flatten_mask = mask.flatten()

        if near_center <= 0: # case for uniform weights
            points = np.random.choice(idx, n_points, p = flatten_mask/flatten_mask.sum())
        else:  # case for higher weights for points near the center
            distance = (distance_transform_edt(mask)**near_center).flatten()
            points = np.random.choice(idx, n_points, p=(distance/distance.sum()))

        x, y = np.unravel_index(points, mask.shape)
        points = np.stack((y, x), axis=1)
    
        return points

    def _get_negative_points(self, mask_path : str, index : int, n_neg_points : int, inside_box : bool = False, 
                             zoom_out : float = 1.0, random_box_shift : int = 0, box_around_mask : bool = False, 
                             mask_prompt_type : str = 'truth', opened_mask : np.ndarray = None):
        """Get n_neg_points points outside the mask
        n_neg_points: int, number of negative points to get
        inside_box: bool, if True, negative points will be inside the bounding box of the mask (but still outside the mask),
        Please refer to _get_box as the other parameters correspond to the same parameters in _get_box."""
        if mask_path:
            mask = plt.imread(mask_path)
        else:
            mask = opened_mask

        idx = np.arange(mask.shape[0] * mask.shape[1])
        flatten_mask = mask.flatten()
        flatten_mask = np.where(flatten_mask > 0, 1, 0)

        if inside_box:
            dilation = 100 # dilation is applied to both box_mask and mask to avoid points near the true mask

            if self.prompts['box'][index] is None:
                x_min, y_min, x_max, y_max = self._get_box(mask_path, index, zoom_out, random_box_shift, box_around_mask, mask_prompt_type, opened_mask = opened_mask)
            else:
                x_min, y_min, x_max, y_max = self.prompts['box'][index]

            box_mask = np.ones_like(mask) * 255
            box_mask[y_min:y_max, x_min:x_max] = 0

            kernel_box = getStructuringElement(MORPH_ELLIPSE, (int(dilation * 1.2), int(dilation * 1.2)))
            kernel_item = getStructuringElement(MORPH_ELLIPSE, (dilation, dilation))

            flatten_mask = erode(box_mask, kernel_box).flatten() + dilate(mask, kernel_item).flatten()
            flatten_mask = np.where(flatten_mask > 0, 1, 0)

        probabilities = (1 - flatten_mask)/((1 - flatten_mask).sum()) # use probabilities of the 0 in the mask

        points = np.random.choice(idx, n_neg_points, p=probabilities)
        x, y = np.unravel_index(points, mask.shape)
        points = np.stack((y, x), axis=1)

        return points
    
    def _get_box(self, mask_path : str, index : int, zoom_out : float=1.0, random_box_shift : int = 0, 
                 box_around_mask : bool = False, mask_prompt_type : str = 'truth', opened_mask : np.ndarray = None) -> tuple[int, int, int, int]:
        """Get a box from the mask
        zoom_out: float, factor to zoom out the box. Add a margin around the mask.
        random_box_shift: int, if greater than 0, the bounding box corners will be shifted randomly by a value between -random_box_shift and random_box_shift
        box_around_mask: bool, if True, the box will be around the input mask."""
        assert zoom_out > 0, 'Zoom out factor must be greater than 0'

        if box_around_mask:
            if self.prompts['mask'][index] is None:
                mask = self._get_mask(mask_path, index, mask_prompt_type = mask_prompt_type, opened_mask = opened_mask)
            else:
                mask = self.prompts['mask'][index]

            mask = resize(mask, (IMG_RESOLUTION, IMG_RESOLUTION), interpolation = INTER_NEAREST)
        else:
            if mask_path:
                mask = plt.imread(mask_path)
            else:
                mask = opened_mask

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
            shifts = np.random.randint(-random_box_shift, random_box_shift, 4)

        x_min = max(0, x_min - h_padding + shifts[0])
        x_max = min(mask.shape[0], x_max + h_padding + shifts[1])

        y_min = max(0, y_min - v_padding + shifts[2])
        y_max = min(mask.shape[1], y_max + v_padding + shifts[3])

        return int(x_min), int(y_min), int(x_max), int(y_max)

    def _get_mask(self, mask_path : str, index : int, mask_prompt_type : str = 'truth', opened_mask : np.ndarray = None) -> np.ndarray:
        """Get the mask from the mask path/
        mask_prompt_type: str, type of mask to use for automatic annotation. Can be 'truth' or 'morphology' or 'scribble'  or 'loose_dilation'. Default is 'truth'.
        'truth': the mask is the truth mask, 'morphology': the mask is the truth mask dilated by a 40x40 kernel, 'scribble': the mask is a simulated scribbled mask by the user."""
        if mask_path:
            mask = resize(plt.imread(mask_path), (INPUT_MASK_RESOLUTION, INPUT_MASK_RESOLUTION), interpolation = INTER_NEAREST)
        else:
            mask = resize(opened_mask, (INPUT_MASK_RESOLUTION, INPUT_MASK_RESOLUTION), interpolation = INTER_NEAREST)

        if mask_prompt_type == 'truth':
            return np.where(mask > 0, 1, 0)
        
        if mask_prompt_type == 'morphology':
            return self._get_mask_morphology(mask)
        
        if mask_prompt_type == 'scribble':
            return self._get_mask_scribble(mask)
        
        if mask_prompt_type == 'loose_dilation':
            return self._get_mask_loose_dilation(mask)
        
        raise ValueError('mask_prompt_type must be truth, morphology, loose_dilation or scribble')
    
    def _get_mask_morphology(self, mask : np.ndarray) -> np.ndarray:
        """Get a mask dilated from the ground truth mask."""
        n = 40
        kernel = getStructuringElement(MORPH_ELLIPSE, (n, n))

        return dilate(mask, kernel=kernel)
    
    def _get_mask_scribble(self, mask : np.ndarray) -> np.ndarray:
        """Get the scribble mask from the mask. n points are chosen on each side (left and right) of the mask. Then
        a line is drawn between corresponding left and right points to simulate human drawn scribbles.
        mask: np.ndarray, the mask to scribble on.
        """
        n = 8
        h_sum = mask.sum(axis=1)
        y_min = np.argmax(h_sum > 0) # index of the first true in the array, meaning first non 0 row
        y_max = len(h_sum) - np.argmax(h_sum[::-1] > 0) # index of the last row

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
    
    def _get_mask_loose_dilation(self, mask : np.ndarray, looseness : int = 20, noise_level : int = 20, iterations : int = 1) -> np.ndarray:
        """Get a mask with a loose dilation strategy, by adding noise to the contour, applying convexHull and then dilating the mask."""
        mask_ = np.where(mask > 0, 1, 0).astype(np.uint8)  # converting mask to binary (0 and 1)
        new_mask = np.zeros_like(mask_, dtype = np.uint8)
        
        contours, _ = findContours(mask_, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        for contour in contours:
            noisy_contour = []
            
            for point in contour:
                x, y = point[0]
                noisy_x = x + np.random.randint(-noise_level, noise_level)
                noisy_y = y + np.random.randint(-noise_level, noise_level)
                
                noisy_contour.append([[noisy_x, noisy_y]])
            
            noisy_contour = np.array(noisy_contour, dtype = np.int32)
            noisy_contour = convexHull(noisy_contour)
            drawContours(new_mask, [noisy_contour], -1, 1, thickness = FILLED)
        
        kernel = np.ones((looseness, looseness), np.uint8)
        new_mask = dilate(new_mask, kernel, iterations = iterations)
        
        return new_mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx : int) -> tuple:
        img = plt.imread(self.images[idx])
        mask = plt.imread(self.masks[idx])

        prompt = {'points':self.prompts['points'][idx], 'box':self.prompts['box'][idx], 'neg_points':self.prompts['neg_points'][idx], 'mask':self.prompts['mask'][idx]}

        return img, np.where(mask > 0, 1, 0), prompt # the returned mask is binary


class SAMDataset(AbstractSAMDataset):
    """Prepare a dataset for segmentation by Segment Anything Model. Checkout AbstractSAMDataset for more information"""

    def __init__(self, root : str, transform = None, use_img_embeddings : bool = False,
                 prompt_type : dict = {'points':False, 'box': False, 'neg_points':False, 'mask':False}, 
                 n_points : int = 1, n_neg_points : int = 1, zoom_out : float = 1.0, verbose : bool = False, 
                 random_state : int = None, to_dict : bool = True, is_sam2_prompt : bool = False, neg_points_inside_box : bool = False, 
                 points_near_center : float = -1, random_box_shift : int = 0, mask_prompt_type : str = 'truth', 
                 box_around_mask : bool = False, is_embedding_saving : bool = False):
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
        is_sam2_prompt: whether the prompt must be formatted for sam or sam2
        neg_points_inside_box: bool, if True, negative points will be inside the bounding box of the mask
        points_near_center: float, if greater than 0, points will be more likely to be near the center of the mask
        random_box_shift: int, if greater than 0, the bounding box corners will be shifted randomly by a value between -random_box_shift and random_box_shift
        mask_prompt_type: str, type of mask to use for automatic annotation. Can be 'truth' or 'morphology' or 'scribble' or 'loose_dilation'. Default is 'truth'.
        box_around_mask: bool, if True, the box will be around the input mask.
        is_embedding_saving: whether the dataset is only used to save the embeddings (in order to get both image and prompts embeddings at the same time)
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
        self.is_sam2_prompt = is_sam2_prompt
        self.zoom_out = zoom_out
        self.is_embedding_saving = is_embedding_saving

        if random_state is not None:
            torch.manual_seed(random_state) # wrong because set seed in torch and use numpy for random ...

        if self.verbose:
            print('Loading images and masks paths...')

        self._load_data()
        self._load_prompt(zoom_out=zoom_out, n_points=n_points, n_neg_points=n_neg_points, inside_box=neg_points_inside_box, near_center=points_near_center, random_box_shift=random_box_shift, mask_prompt_type=mask_prompt_type, box_around_mask=box_around_mask)
        
        if self.use_img_embeddings:
            self._load_img_embeddings()

        if self.verbose:
            print('Done!')

    def _load_data(self):
        """Load images and masks"""
        for f in os.listdir(self.root + 'processed/'):
            for g in os.listdir(self.root + 'processed/' + f):
                if g.endswith('.jpg'):
                    if 'mask' in g:
                        self.masks.append(self.root + 'processed/' + f + '/' + g)
                    else:
                        self.images.append(self.root + 'processed/' + f + '/' + g)

    def _load_img_embeddings(self):
        """Load image embeddings"""
        self.img_embeddings = []
        for f in os.listdir(self.root + 'img_embeddings/'):
            self.img_embeddings.append(torch.load(self.root + 'img_embeddings/' + f).to('cpu'))

    def __getitem__(self, idx : int) -> tuple:
        img = plt.imread(self.images[idx])
        mask = plt.imread(self.masks[idx])
        prompt = {'points':self.prompts['points'][idx], 'box':self.prompts['box'][idx], 'neg_points':self.prompts['neg_points'][idx], 'mask':self.prompts['mask'][idx]}
    
        if self.use_img_embeddings:
            return to_dict(self.img_embeddings[idx], prompt, use_img_embeddings = self.use_img_embeddings, is_sam2_prompt = self.is_sam2_prompt), np.where(mask > 0, 1, 0)
    
        if self.transform:
            img, mask = self.transform(img, mask)

        if self.to_dict:
            if self.is_embedding_saving:
                prompt_copy = deepcopy(prompt)
                return to_dict(img, prompt, is_sam2_prompt = self.is_sam2_prompt), np.where(mask > 0, 1, 0), prompt_copy
            
            else:
                return to_dict(img, prompt, is_sam2_prompt = self.is_sam2_prompt), np.where(mask > 0, 1, 0)

        return img, np.where(mask > 0, 1, 0), prompt
    

class AugmentedSamDataset(SAMDataset):
    """Prepare an augmented dataset for segmentation by Segment Anything Model. 
    Checkout AbstractSAMDataset for more information.
    Each samples exist in multiple versions with different prompts."""

    def __init__(self, root : str, use_img_embeddings : bool = True,
                 n_points : int = 1, n_neg_points : int = 1, zoom_out : float = 1.0, verbose : bool = False, 
                 random_state : int = None, to_dict : bool = True, is_sam2_prompt : bool = False, 
                 random_box_shift : int = 20, mask_prompt_type : str = 'truth',
                 load_on_cpu : bool = False, filter_files : callable = None):
        '''Initialize SAMDataset class.
        root: str, path to the dataset directory
        use_img_embeddings: bool, if True, the model will use image embeddings instead of the images
        n_points: int, number of points to use for automatic annotation if prompt is 'points'.
        n_neg_points: int, number of negative points to use for automatic annotation if prompt is 'neg_points'.
        zoom_out: float, factor to zoom out the bounding box. Add a margin around the mask.
        verbose: bool, if True, print progress messages
        random_state: int, random state for reproducibility
        to_dict: bool, if True, the __getitem__ method will return a dictionary with the image, the prompt and the mask. If False, it will return the image, the mask and the prompt.
        is_sam2_prompt: whether the prompt must be formatted for sam or sam2        
        random_box_shift: int, if greater than 0, the bounding box corners will be shifted randomly by a value between -random_box_shift and random_box_shift
        mask_prompt_type: str, type of mask to use for automatic annotation. Can be 'truth' or 'morphology' or 'scribble' or 'loose_dilation'. Default is 'truth'.
        load_on_cpu: bool, if True, the entire dataset will be loaded on the CPU RAM. Data loading will be faster but will consume more memory. Default: False
        filter_files: callable, function to filter the files from the dataset.
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
                         to_dict=to_dict, is_sam2_prompt = is_sam2_prompt, 
                         neg_points_inside_box=neg_points_inside_box, 
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
        
    def __getitem__(self, idx : int) -> tuple:
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
            return to_dict(self.img_embeddings[img_idx], prompt, use_img_embeddings = self.use_img_embeddings, is_sam2_prompt = self.is_sam2_prompt), np.where(mask > 0, 1, 0)
        
        if self.transform:
            img, mask = self.transform(img, mask)

        if self.to_dict:
            return to_dict(img, prompt, is_sam2_prompt = self.is_sam2_prompt), np.where(mask > 0, 1, 0)
        
        return img, np.where(mask > 0, 1, 0), prompt
    
    def _load_data(self):
        """Load images and masks. Allows to filter files."""
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
        """Load image embeddings"""
        self.img_embeddings = []
        for f in os.listdir(self.root + 'img_embeddings/'):
            if self.filter_files is not None:
                if not self.filter_files(f):
                    continue

            self.img_embeddings.append(torch.load(self.root + 'img_embeddings/' + f).to('cpu'))

    def __len__(self):
        return len(self.images) * 4


class SamDatasetFromFiles(AbstractSAMDataset):
    """Prepare a dataset for segmentation by Segment Anything Model. Checkout AbstractSAMDataset for more information"""

    def __init__(self, root : str, transform = None, use_img_embeddings : bool = False,
                 prompt_type : dict = {'points':False, 'box': False, 'neg_points':False, 'mask':False}, 
                 n_points : int = 1, n_neg_points : int = 1, zoom_out : float = 1.0, verbose : bool = False, 
                 random_state : int = None, to_dict : bool = True, is_sam2_prompt : bool = False, neg_points_inside_box : bool = False, 
                 points_near_center : float = -1, random_box_shift : int = 0, mask_prompt_type : str = 'truth', 
                 box_around_mask : bool = False, filter_files : callable = None, load_on_cpu : bool = False,
                 generate_prompt_on_get : bool = False, is_combined_embedding : bool = False, is_embedding_saving : bool = False):
        """
        Constructor of the SamDatasetFromFiles. It is highly adviced to use this dataset when finetuning.

        Args:
            root (str): path to the dataset directory
            transform (callable, optional): transform function for data augmentation on the images. Defaults to None.
            use_img_embeddings (bool, optional): if True, the model will use image embeddings instead of the images. Defaults to False.
            prompt_type (dict, optional): prompt types to consider. Defaults to {'points':False, 'box': False, 'neg_points':False, 'mask':False}.
            n_points (int, optional): number of points to use for automatic annotation if prompt is 'points'. Defaults to 1.
            n_neg_points (int, optional): number of negative points to use for automatic annotation if prompt is 'neg_points'. Defaults to 1.
            zoom_out (float, optional): factor to zoom out the bounding box. Add a margin around the mask. Defaults to 1.0.
            verbose (bool, optional): if True, print progress messages. Defaults to False.
            random_state (int, optional): random state for reproducibility. Defaults to None.
            to_dict (bool, optional): if True, the __getitem__ method will return a dictionary with the image, the prompt and the mask. 
                                    If False, it will return the image, the mask and the prompt. Defaults to True.
            is_sam2_prompt (bool, optional): whether the prompt must be formatted for sam or sam2. Defaults to False.
            neg_points_inside_box (bool, optional): if True, negative points will be inside the bounding box of the mask. Defaults to False.
            points_near_center (float, optional): if greater than 0, points will be more likely to be near the center of the mask. Defaults to -1.
            random_box_shift (int, optional): if greater than 0, the bounding box corners will be shifted randomly by a value between -random_box_shift and random_box_shift. Defaults to 0.
            mask_prompt_type (str, optional): type of mask to use for automatic annotation. Can be 'truth' or 'morphology' or 'scribble' or 'loose_dilation'. Default is 'truth'.
            box_around_mask (bool, optional): if True, the box will be around the input mask. Defaults to False.
            filter_files (callable, optional): callable, function to filter the files from the dataset. Defaults to None.
            load_on_cpu (bool, optional): if True, the entire dataset will be loaded on the CPU RAM. Data loading will be faster but will consume more memory. Defaults to False.
            generate_prompt_on_get (bool, optional): if this is True, the prompt will be generated randomly when calling get_item instead of preloaded. Defaults to False.
            is_combined_embedding (bool, optional): used for HistoSAM to combine the embeddings of the 2 encoders. Defaults to False.
            is_embedding_saving (bool, optional): whether the dataset is only used to save the embeddings (in order to get both image and prompts embeddings at the same time). Defaults to False.
        """    
        print('Creating dataset...')
        self.root = root
        self.transform = transform
        self.use_img_embeddings = use_img_embeddings
        self.is_combined_embedding = is_combined_embedding
        self.load_on_cpu = load_on_cpu
        self.generate_prompt_on_get = generate_prompt_on_get
        self.is_embedding_saving = is_embedding_saving

        self.images = []
        self.masks = []
        self.img_embeddings = []
        self.prompts = {
            'points': [], 
            'box': [], 
            'neg_points': [],
            'mask': []
        }
        self.prompt_embeddings = []

        self.prompt_type = prompt_type
        self.verbose = verbose

        self.n_points = n_points
        self.n_neg_points = n_neg_points
        self.near_center = points_near_center
        self.inside_box = neg_points_inside_box
        self.random_box_shift = random_box_shift
        self.mask_prompt_type = mask_prompt_type
        self.box_around_mask = box_around_mask

        self.to_dict = to_dict
        self.is_sam2_prompt = is_sam2_prompt
        self.zoom_out = zoom_out

        self.filter_files = filter_files

        if random_state is not None:
            torch.manual_seed(random_state)

        if self.verbose:
            print('Loading data...')

        self._load_data()
        self.PROMPT_COMBINATIONS_LIST = {
            'single': [['points'], ['box'], ['neg_points'], ['mask']],
            'pair': [['box', 'points'], ['box', 'neg_points'], ['points', 'neg_points'], ['mask', 'points'], ['mask', 'neg_points']],
            'triple': [['box', 'points', 'neg_points'], ['mask', 'points', 'neg_points']],
            'all': [['points', 'box', 'neg_points', 'mask']]
        }

        self.prompt_combinations = self.select_combinations(self.prompt_type, self.PROMPT_COMBINATIONS_LIST)
        self.nb_combinations = len(self.prompt_combinations)

        if self.generate_prompt_on_get:
            self.prompts = {
                'points': [None for _ in range(len(self.images))], 
                'box': [None for _ in range(len(self.images))], 
                'neg_points': [None for _ in range(len(self.images))],
                'mask': [None for _ in range(len(self.images))]
            }

        print('Selected prompt combinations: ')
        for c in self.prompt_combinations:
            print(c)

        if self.verbose:
            print('Done!')

    def select_combinations(self, prompt_type : dict, combination_list : dict):
        """
        Function to select the prompt combinations to be considered during training.

        Args:
            prompt_type (dict): the prompt types that need to be considered.
            combination_list (dict): dict of all possible combinations.

        Returns:
            list of prompt combinations to consider.
        """
        active_prompts = [key for key, value in prompt_type.items() if value]
        nb_active = len(active_prompts)

        valid_combinations = []
        if nb_active == 1:
            for comb in combination_list['single']:
                if comb[0] in active_prompts:
                    valid_combinations.append(comb)

        elif nb_active == 2:
            for comb in combination_list['single']:
                if comb[0] in active_prompts:
                    valid_combinations.append(comb)

            for comb in combination_list['pair']:
                if all(prompt in active_prompts for prompt in comb):
                    valid_combinations.append(comb)

        elif nb_active == 3:
            for comb in combination_list['single']:
                if comb[0] in active_prompts:
                    valid_combinations.append(comb)

            for comb in combination_list['triple']:
                if all(prompt in active_prompts for prompt in comb):
                    valid_combinations.append(comb)

        elif nb_active == 4:
            valid_combinations.append(['box'])
            valid_combinations.append(['mask'])

            for comb in combination_list['triple']:
                valid_combinations.append(comb)

            valid_combinations.append(combination_list['all'][0])

        return valid_combinations

    def _load_data(self):
        """Load images and masks. Allows to filter files."""
        nb_imgs = 0
        nb_filtered_out = 0
        file_counts = {}

        for u, f in enumerate(os.listdir(self.root + 'processed/')):
            nb_imgs += 1
            if self.verbose and u % 100 == 0:
                print(f"Progress: {u} iterations")

            if self.filter_files is not None:
                if not self.filter_files(f):
                    nb_filtered_out += 1
                    continue

            i_value = f.split('_')[0]
            if i_value not in file_counts:
                file_counts[i_value] = 0

            file_counts[i_value] += 1

            current_path = self.root + 'processed/' + f
            for g in os.listdir(current_path):
                full_path = os.path.join(current_path, g)

                if g.endswith('.jpg'):
                    if 'mask' in g:
                        self.masks.append(full_path if not self.load_on_cpu else plt.imread(full_path))
                    else:
                        self.images.append(full_path if not self.load_on_cpu else plt.imread(full_path))

                if self.use_img_embeddings:
                    if (g == 'img_embedding.pt' and not self.is_sam2_prompt) or (g == 'sam2_img_embedding.pt' and self.is_sam2_prompt):
                        if self.load_on_cpu:
                            loaded_data = torch.load(full_path)

                            if isinstance(loaded_data, dict):
                                if not self.is_combined_embedding:
                                    for key, value in loaded_data.items():
                                        if key == 'image_embed':
                                            loaded_data[key] = value.squeeze(0).to('cpu')

                                        elif key == 'high_res_feats':
                                            for i, v in enumerate(value):
                                                loaded_data[key][i] = v.squeeze(0).to('cpu')

                                        else:
                                            raise ValueError(f'Key {key} not supported')
                                        
                                else:
                                    for key, value in loaded_data.items():
                                        loaded_data[key] = value.to('cpu')

                                self.img_embeddings.append(loaded_data)

                            else:
                                self.img_embeddings.append(loaded_data.to('cpu'))

                        else:
                            self.img_embeddings.append(full_path)

                if g == 'prompt.pt' and self.generate_prompt_on_get == False:
                    if self.load_on_cpu:
                        temp_prompt = torch.load(full_path)

                        self.prompts['points'].append(temp_prompt['points'])
                        self.prompts['neg_points'].append(temp_prompt['neg_points'])
                        self.prompts['box'].append(temp_prompt['box'])
                        self.prompts['mask'].append(temp_prompt['mask'])
                    else:
                        self.prompt_embeddings.append(full_path)

        if self.verbose:
            print(f'Initial number of images: {nb_imgs}')
            print(f'Number of filtered out: {nb_filtered_out}')

            print(f'Number of images in dataset: {len(self.images)}')
            print(f'Number of masks in dataset: {len(self.masks)}')

            print(f'Number of point prompts: {len(self.prompts["points"])}')
            print(f'Number of neg point prompts: {len(self.prompts["neg_points"])}')
            print(f'Number of box prompts: {len(self.prompts["box"])}')
            print(f'Number of mask prompts: {len(self.prompts["mask"])}')

            print('Files analysis:')
            for key, value in file_counts.items():
                print(f'Dataset key: {key}, count: {value}')

    def __getitem__(self, idx : int) -> tuple:
        img_idx = idx % len(self.images)
        prompt_idx = idx // len(self.images)

        prompt = {'points' : None, 'box' : None, 'neg_points' : None, 'mask' : None}

        if self.load_on_cpu:
            img = self.images[img_idx]
            mask = self.masks[img_idx]

            if self.generate_prompt_on_get == False:
                for key in self.prompt_combinations[prompt_idx]:
                    prompt[key] = self.prompts[key][img_idx]

            else:
                for key in self.prompt_combinations[prompt_idx]:
                    if key == 'points':
                        prompt[key] = self._get_points(mask_path = None, index = -1, n_points = self.n_points, near_center = self.near_center, opened_mask = mask)

                    elif key == 'box':
                        prompt[key] = self._get_box(mask_path = None, index = img_idx, zoom_out = self.zoom_out, random_box_shift = self.random_box_shift, 
                                                    box_around_mask = self.box_around_mask, mask_prompt_type = self.mask_prompt_type, opened_mask = mask)

                    elif key == 'neg_points':
                        prompt[key] = self._get_negative_points(mask_path = None, index = img_idx, n_neg_points = self.n_neg_points, inside_box = self.inside_box, 
                             zoom_out = self.zoom_out, random_box_shift = self.random_box_shift, box_around_mask = self.box_around_mask, 
                             mask_prompt_type = self.mask_prompt_type, opened_mask = mask)

                    else: # key = mask
                        prompt[key] = self._get_mask(mask_path = None, index = -1, mask_prompt_type = self.mask_prompt_type, opened_mask = mask)

            if self.use_img_embeddings:
                img_embedding = self.img_embeddings[img_idx]

        else:
            img = plt.imread(self.images[img_idx])
            mask = plt.imread(self.masks[img_idx])

            if self.generate_prompt_on_get == False:
                loaded_prompt = torch.load(self.prompt_embeddings[img_idx], map_location = 'cpu')

                for key in self.prompt_combinations[prompt_idx]:
                    prompt[key] = loaded_prompt[key]

            else:
                for key in self.prompt_combinations[prompt_idx]:
                    if key == 'points':
                        prompt[key] = self._get_points(mask_path = self.masks[img_idx], index = -1, n_points = self.n_points, near_center = self.near_center)

                    elif key == 'box':
                        prompt[key] = self._get_box(mask_path = self.masks[img_idx], index = img_idx, zoom_out = self.zoom_out, random_box_shift = self.random_box_shift, 
                                                    box_around_mask = self.box_around_mask, mask_prompt_type = self.mask_prompt_type)

                    elif key == 'neg_points':
                        prompt[key] = self._get_negative_points(mask_path = self.masks[img_idx], index = img_idx, n_neg_points = self.n_neg_points, inside_box = self.inside_box, 
                             zoom_out = self.zoom_out, random_box_shift = self.random_box_shift, box_around_mask = self.box_around_mask, mask_prompt_type = self.mask_prompt_type)

                    else: # key = mask
                        prompt[key] = self._get_mask(mask_path = self.masks[img_idx], index = -1, mask_prompt_type = self.mask_prompt_type)

            if self.use_img_embeddings:
                img_embedding = torch.load(self.img_embeddings[img_idx], map_location = 'cpu')
                if isinstance(img_embedding, dict):
                    if not self.is_combined_embedding:
                        for key, value in img_embedding.items():
                            if key == 'image_embed':
                                img_embedding[key] = value.squeeze(0).to('cpu')

                            elif key == 'high_res_feats':
                                for i, v in enumerate(value):
                                    img_embedding[key][i] = v.squeeze(0).to('cpu')

                            else:
                                raise ValueError(f'Key {key} not supported')
                            
                    else:
                        for key, value in img_embedding.items():
                            img_embedding[key] = value.to('cpu')

        if self.use_img_embeddings:
            return to_dict(img_embedding, prompt, use_img_embeddings = self.use_img_embeddings, is_sam2_prompt = self.is_sam2_prompt), np.where(mask > 0, 1, 0)
    
        if self.transform:
            img, mask = self.transform(img, mask)

        if self.to_dict:
            if self.is_embedding_saving:
                prompt_copy = deepcopy(prompt)
                return to_dict(img, prompt, is_sam2_prompt = self.is_sam2_prompt), np.where(mask > 0, 1, 0), prompt_copy

            else:
                return to_dict(img, prompt, is_sam2_prompt = self.is_sam2_prompt), np.where(mask > 0, 1, 0)
        
        return img, np.where(mask > 0, 1, 0), prompt
    
    def __len__(self):
        return len(self.images) * self.nb_combinations # * by nb combinations because this is a data augmentation trick


def filter_dataset(file_name, datasets : list):
    """Filter datasets to keep only the ones.
    datasets: List[str], list of boolean values to filter, True if the dataset is to keep, False otherwise."""
    file_name_dataset = file_name.split('_')[0]

    return datasets[int(file_name_dataset)]