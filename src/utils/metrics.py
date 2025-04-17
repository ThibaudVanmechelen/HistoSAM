import cv2
import numpy as np


def compute_compactness(mask : np.ndarray):
    """
    Function to compute the compactness of a mask.
    Compactness is a shape descriptor that measures how closely packed a shape is.

    Args:
        mask (np.ndarray): the mask.

    Returns:
        float: the compactness score. 0 if no valid contour is found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0

    perimeter = cv2.arcLength(contours[0], True)
    if perimeter == 0:
        return 0
    
    area = cv2.contourArea(contours[0])
    
    return (4 * np.pi * area) / (perimeter ** 2)


def compute_solidity(mask : np.ndarray):
    """
    Function to compute the solidity of a mask.
    Solidity is the ratio of the object area to the area of its convex hull.

    Args:
        mask (np.ndarray): the mask.

    Returns:
        float: The solidity score. 0 if no valid contour or convex hull is found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0

    hull = cv2.convexHull(contours[0])
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0

    area = cv2.contourArea(contours[0])

    return area / hull_area


def compute_perimeter_smoothness_ratio(mask : np.ndarray):
    """
    Function to compute the perimeter smoothness ratio.
    This ratio compares the raw perimeter of a shape to that of its polygonal approximation.

    Args:
        mask (np.ndarray): the mask.

    Returns:
       float: The ratio of original perimeter to smoothed (approximated) perimeter
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0

    perimeter = cv2.arcLength(contours[0], True)

    poly_contour = cv2.approxPolyDP(contours[0], 0.02 * perimeter, True)
    poly_perimeter = cv2.arcLength(poly_contour, True)

    if poly_perimeter == 0:
        return 0
    
    return perimeter / poly_perimeter


def count_connected_components(mask : np.ndarray):
    """
    Counts the number of connected components in a mask, excluding the background.

    Args:
        mask (np.ndarray): the mask.

    Returns:
        int: The number of connected components (excluding the background).
    """
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity = 8)
    return num_labels - 1  # because need to exclude the background


def compute_size_retention(original_mask : np.ndarray, postprocessed_mask : np.ndarray):
    """
    Function to compute the size retntion ratio of the postprocessed mask.
    This measures how much of the original object area remains after postprocessing.

    Args:
        original_mask (np.ndarray): The original binary mask.
        postprocessed_mask (np.ndarray): The binary mask after postprocessing.

    Returns:
        float: The ratio of postprocessed area to original area. Returns 0 if original area is zero.
    """
    original_area = np.sum(original_mask > 0)
    postprocessed_area = np.sum(postprocessed_mask > 0)
    
    if original_area == 0:
        return 0
    
    return postprocessed_area / original_area