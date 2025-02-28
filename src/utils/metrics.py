import cv2
import numpy as np


def compute_compactness(mask : np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0

    perimeter = cv2.arcLength(contours[0], True)
    if perimeter == 0:
        return 0
    
    area = cv2.contourArea(contours[0])
    
    return (4 * np.pi * area) / (perimeter ** 2)


def compute_solidity(mask : np.ndarray):
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
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity = 8)
    return num_labels - 1  # because need to exclude the background


def compute_size_retention(original_mask : np.ndarray, postprocessed_mask : np.ndarray):
    original_area = np.sum(original_mask > 0)
    postprocessed_area = np.sum(postprocessed_mask > 0)
    
    if original_area == 0:
        return 0
    
    return postprocessed_area / original_area