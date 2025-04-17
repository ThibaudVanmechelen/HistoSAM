import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels


def post_process_segmentation_mask(mask : np.ndarray, opening_kernel_size : int = 10,
                                   closing_kernel_size : int = 20, blur_size : int = 21,
                                   do_gaussian_blur : bool = False):
    """
    Function to apply post processing to the segmentation mask returned by the model.
    This function first performs an opening to remove small noises followed by a closing
    to thicken the mask. It can optionnally also apply gaussian blur.

    Args:
        mask (np.ndarray): the mask to postprocess
        opening_kernel_size (int, optional): size of the opening kernel. Defaults to 10.
        closing_kernel_size (int, optional): size of the closing kernel. Defaults to 20.
        blur_size (int, optional): size of the blur. Defaults to 21.
        do_gaussian_blur (bool, optional): whether to apply blur. Defaults to False.

    Returns:
        (np.ndarray): the postprocessed mask.
    """
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)

    opening_kernel = np.ones((opening_kernel_size, opening_kernel_size), np.uint8)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)

    closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
    temp_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, closing_kernel)

    if do_gaussian_blur:
        _, temp_mask = cv2.threshold(cv2.GaussianBlur(temp_mask, (blur_size, blur_size), 0), 127, 255, cv2.THRESH_BINARY)

    temp_mask = filter_mask_by_size(temp_mask)

    return temp_mask


def filter_mask_by_size(mask : np.ndarray, area_thresh_percentage : float = 0.1):
    """
    Function to clean the mask, this function cleans the mask by removing connected area which are less than 10% of the biggest part of the mask

    Args:
        mask (np.ndarray): the mask to clean.
        area_thresh_percentage (float, optional): The threshold in % at which we remove the small elements. Defaults to 0.1.

    Returns:
        (np.ndarray): the cleaned mask.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity = 8)
    areas = stats[:, cv2.CC_STAT_AREA]

    if len(areas) <= 1: # there is just background here, empty mask
        return np.zeros_like(mask)

    max_area = np.max(areas[1:]) # max area is the principal mask
    min_area = area_thresh_percentage * max_area # min area that an element should have

    filtered_mask = np.zeros_like(mask)
    for label in range(1, num_labels):
        if areas[label] >= min_area:
            filtered_mask[labels == label] = 255

    return filtered_mask


def post_process_with_crf(img : np.ndarray, mask : np.ndarray):
    """
    Function to postprocess the mask using crf.

    Args:
        img (np.ndarray): the original image.
        mask (np.ndarray): the mask outputted by SAM.

    Returns:
        (np.ndarray): the postprocessed mask.
    """
    height, width = mask.shape
    mask = (mask > 0).astype(np.uint8)

    dense_crf = dcrf.DenseCRF2D(width, height, 2)  # 2-class segmentation (foreground, background)
    unary = unary_from_labels(mask + 1, 2, gt_prob = 0.7)

    dense_crf.setUnaryEnergy(unary)
    dense_crf.addPairwiseGaussian(sxy = 3, compat = 3)
    dense_crf.addPairwiseBilateral(sxy = 5, srgb = 13, rgbim = img, compat = 5)

    refined_mask = np.argmax(dense_crf.inference(5), axis = 0).reshape(height, width)

    return refined_mask