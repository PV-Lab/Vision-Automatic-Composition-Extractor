# Author: Alexander (Aleks) E. Siemenn <asiemenn@mit.edu>
# Date:   04 March 2023

# import
import cv2
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import ndimage

def segmentation(img, rotate_crop_params, lower_pixel_thresh=50, upper_pixel_thresh=3500):
    '''
    ==========================
    == VISION SEGMENTATION  ==
    ==========================

    Inputs:
        img:                  Path to the RGB image file
        rotate_crop_params:   User-defined dictionary of values used to rotate and crop the image. E.g., {'theta': -0.5, 'x1': 45, 'x2': 830, 'y1': 120, 'y2': 550}
        lower_pixel_thresh:   Elements containing fewer than the specified number of pixels will be removed
        upper_pixel_thresh:   Elements containing more than the specified number of pixels will be removed

    Outputs:
        data:                 An array containing the indices of segmented object
    '''

    # load RGB image
    img = (plt.imread(img)*255).astype('uint8')
    # plot raw image
    plt.imshow(img)
    plt.title('Raw Image')
    plt.show()
    # crop/rotate single band
    img = ndimage.rotate(img, rotate_crop_params['theta'])  # reads image and rotates
    img = img[rotate_crop_params['y1']:rotate_crop_params['y2'],
                 rotate_crop_params['x1']:rotate_crop_params['x2']]  # crops image
    # plot cropped image
    plt.imshow(img)
    plt.title('Cropped Image')
    plt.show()
    # transpose to segment along print direction
    imgT = ndimage.rotate(img, -90)
    imgT = cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY)
    # segment image with watershed
    watershed, edges = watershed_segment(imgT, small_elements_pixels=lower_pixel_thresh,
                                         large_elements_pixels=upper_pixel_thresh)
    # plot segmented image
    fig, ax = plt.subplots()
    ax.imshow(watershed.T, cmap='viridis')
    ax.invert_yaxis()
    plt.title('Segmented Image')
    plt.show()

    return watershed.T, edges.T


# Sub-functions for segmentation()

def segment_on_dt(a, img, threshold):
    '''
    Implements watershed segmentation.

    Inputs:
        a:         Image input
        img:       Threshold binned image
        threshold: RGB threshold value

    Outputs:
        lbl:       Borders of segmented droplets
        wat:       Segmented droplets via watershed
        lab:       Indices of each segmented droplet
    '''
    # estimate the borders of droplets based on known and unknown background + foreground (computed using dilated and erode)
    kernel = np.ones((5, 5), np.uint8)
    border = cv2.dilate(img, None, iterations=1)
    border = border - cv2.erode(border, kernel)
    # segment droplets via distance mapping and thresholding
    dt = cv2.distanceTransform(img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
    _, dt = cv2.threshold(dt, threshold, 255, cv2.THRESH_BINARY)
    # obtain the map of segmented droplets with corresponding indices
    lbl, ncc = ndimage.label(dt)
    lbl = lbl * (255 / (ncc + 1))
    lab = lbl
    # Completing the markers now.
    lbl[border == 255] = 255
    lbl = lbl.astype(np.int32)
    a = cv2.cvtColor(a,cv2.COLOR_GRAY2BGR)  # we must convert grayscale to BGR because watershed only accepts 3-channel inputs
    wat = cv2.watershed(a, lbl)
    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    return 255 - lbl, wat, lab  # return lab, the segmented and indexed droplets

def watershed_segment(image, small_elements_pixels=0, large_elements_pixels=999999):
    '''
    Applies watershed image segmentation to separate droplet pixels from background pixels.

    Inputs:
        image:                   Input droplet image to segment
        small_elements_pixels:   Removes small elements that contain fewer than the specified number of pixels.
        large_elements_pixels:   Removes large elements that contain more than the specified number of pixels.

    Outputs:
        water:                   Watershed segmented droplets
        labs:                    Edges to each watershed segmented droplet
    '''
    RGB_threshold = 0
    img = image.copy()
    img = 255 - img
    _, img_bin = cv2.threshold(img, 0, 255,
                               # threshold image using Otsu's binarization # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
                               cv2.THRESH_OTSU)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
                               np.ones((12, 12), dtype=int))
    img_bin = cv2.erode(img_bin, np.ones((3, 3), np.uint8))
    img_bin = cv2.medianBlur(img_bin, 7)
    result, water, labs = segment_on_dt(a=img, img=img_bin,
                                        threshold=RGB_threshold)  # segment droplets from background and return indexed droplets
    water = cv2.dilate(water.astype('uint8'), np.ones((5, 5), np.uint8))
    water = cv2.medianBlur(water,5)
    # remove small/large elements
    uniq_full, uniq_counts = np.unique(water,
                                       return_counts=True)  # get all unique watershed indices with pixel counts
    large_elements = uniq_full[uniq_counts > large_elements_pixels]  # mask large elements based on number of pixels
    small_elements = uniq_full[uniq_counts < small_elements_pixels] # mask small elements based on number of pixels
    for n in range(len(large_elements)):
        water[water == large_elements[n]] = 0  # remove all large elements
    for n in range(len(small_elements)):
        water[water == small_elements[n]] = 0  # remove all small elements
    return water, labs
