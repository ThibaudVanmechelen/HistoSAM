import cv2
import numpy as np

def post_process_segmentation_mask(mask, kernel_size = 5, border_kernel_size = 15, blur_size = 5, do_convex_hull = False, do_gaussian_blur = False, do_border_closing = False):
    # Mask should be binary with either 0 or 255
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    temp_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)

    if do_border_closing:
        border_kernel = np.ones((border_kernel_size, border_kernel_size), np.uint8)
        temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_CLOSE, border_kernel)

    if do_gaussian_blur:
        temp_mask = cv2.GaussianBlur(temp_mask, (blur_size, blur_size), 0)

    if do_convex_hull:
        contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_mask = np.zeros_like(temp_mask)

        for cnt in contours:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(hull_mask, [hull], -1, (255), thickness = cv2.FILLED)

        return hull_mask

    return temp_mask


def filter_mask_by_size(mask, area_thresh_percentage = 0.1):
    """
    Function to clean the mask, this function cleans the mask by removing connected area which are less than 10% of the biggest part of the mask
    """
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity = 8)
    areas = stats[:, cv2.CC_STAT_AREA]

    max_area = np.max(areas[1:]) # max area is the principal mask
    min_area = area_thresh_percentage * max_area # min area that an element should have

    filtered_mask = np.zeros_like(binary_mask)

    for label in range(1, num_labels):
        if areas[label] >= min_area:
            filtered_mask[labels == label] = 255

    return filtered_mask


def fill_mask(mask):
    """
    Function to fill the mask.
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness = cv2.FILLED)

    return filled_mask