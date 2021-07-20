import cv2 as cv
from PIL import Image
import numpy as np

# TODO (applicable in various places): avoid magic numbers and parametrize
# all variables (namely, threshold, off, inpaint_radius, min_size, max_size

def _find_rect(img):
    """
    Returns the largest black rectangle found in the image, or None if none is found

    Input:
    img: an image (numpy array, opened with opencv)

    Output:
    x, y, w, h: top-left coordinates of the rectangle (x,y), width (w) and height (h)
                or None if no rectangle is found
    """

    threshold = 30

    # convert each channel (R,G,B) of the image to black/white (based on threshold)
    # then, if an image has all 3 channels black, flag that pixel as black (bw is a
    # 2d boolean matrix of pixels -- 1 if black, 0 if white)
    bw = 255*((img<threshold).sum(axis=2)==3).astype(np.uint8)
    
    # findContours will find the contours in a binary image (bw). More info on the 
    # parameters used can be found in the documentation of the function
    _, contours, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    max_area = None
    # Now we go through all of the contours found. For each one, we find the
    # bounding rectangle (i.e. the rectangle that contains the contours). Then, 
    # we select the largest rectangle, as we expect that to be the one we are
    # interested in
    # NOTE: choosing the largest rectangle (i.e. the one with the largest area) does
    # not guarantee finding the correct rectangle (due to possible false positives/negatives)
    # we try to mitigate this problem as shown below
    for cnt in contours:
        a,b,c,d = cv.boundingRect(cnt)
        if not max_area or max_area[2] * max_area[3] < c * d:
            max_area = a,b,c,d
    
    if not max_area:
        return None

    a,b,c,d = max_area
    img_area = c * d / (img.shape[0] * img.shape[1])


    # here we leverage some "domain knowledge": we know that between 10 and 20%
    # of pixels have been removed from the image. If we do not find a rectangle with
    # an area in that range, we are unlikely to have found the right rectangle, 
    # so we just drop it (instead of returning something we know is wrong)
    min_size = .1
    max_size = .2
    if not min_size <= img_area <= max_size:
        return None
    return max_area

def _remove_rect(img):
    """
    Restore the pixels of the black rectangle using
    some interpolation.

    Input:
    img: input image (numpy array)

    Output:
    out_img: the same image as img, but with the black rectangle restored (if no
             rectangle is detected, img is returned)
    """

    off = 2 # add border of size `off` around the rectangle found (if == 0, it doesn't work quite well)
    inpaint_radius = 3
    flags = cv.INPAINT_TELEA # or _NS
    
    res = _find_rect(img) # find largest black(ish -- see "threshold") rectangle
    if not res:
        # res == None, no rectangles detected -- return img
        return img

    a,b,c,d = res
    # mask contains 0's everywhere and 1's (255) where the detected rectangle is
    # (plus some offset to make the rectangle just slightly larger and improve the
    # performance of inpaint()
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[max(b-off,0):min(b+d+off,mask.shape[0]), max(a-off,0):min(a+c+off,mask.shape[1])] = 255
    
    # using opencv's inpaint to restore the black rectangle from the neighboring
    # pixels (in other words, we interpolate the missing part of the image)
    # (more info here https://docs.opencv.org/4.5.2/df/d3d/tutorial_py_inpainting.html)
    dst = cv.inpaint(img, mask, inpaint_radius, flags)
    return dst


# A class for applyging the "remove rectangle" transformation and
# returning a PIL image
class RemoveRectangle:
    def __init__(self):
        pass
    
    def __call__(self, x):
        return Image.fromarray(_remove_rect(np.array(x)))
