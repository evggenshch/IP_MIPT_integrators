import numpy as np
import copy
from skimage import color
from skimage import io
import random
import math
from sklearn.metrics import mean_squared_error

SRGB_BORDER = 0.04045
RGB_BORDER = 0.0031308

#SRGB_RGB_MATRIX = np.array([[0.18048079, 0.07219232, 0.95053215], \
#                            [0.35758434, 0.71516868, 0.11919478], \
#                            [0.41239080, 0.21263901, 0.01933082]])
#RGB_SRGB_MATRIX = np.array([[], [], []])

#   Define custom gamma correction

def l2_metric(img_prev, img_next):

    return mean_squared_error(img_next, img_prev)

def from_srgb_to_rgb(srgb_img):


   # rgb_img = (copy.deepcopy(srgb_img).astype(np.float32) / 255)
    rgb_img = (copy.deepcopy(srgb_img))

    for row, cur_row in enumerate(rgb_img):
        for column, cur_column in enumerate(cur_row):
            for channel, cur_channel in enumerate(cur_column):
                cur_val = np.float32(srgb_img[row][column][channel]) #/ 255)
            #    rgb_img[row][column][channel] = cur_val ** 2.2
                rgb_img[row][column][channel] =  ((25 * cur_val / 323) if cur_val <= SRGB_BORDER else \
                                                      ((200 * cur_val + 11) / 211) ** (12 / 5))
               # rgb_img[row][column][:] = SRGB_RGB_MATRIX.dot(np.transpose(rgb_img[row][column][:]))


  #  rgb_img = (rgb_img * 255).astype(np.uint8)
    return rgb_img

def from_rgb_to_srgb(rgb_img):

   # srgb_img = (copy.deepcopy(rgb_img).astype(np.float32) / 255)
    srgb_img = copy.deepcopy(rgb_img)

    for row, cur_row in enumerate(srgb_img):
        for column, cur_column in enumerate(cur_row):
            for channel, cur_channel in enumerate(cur_column):
                cur_val = np.float32(rgb_img[row][column][channel])# / 255)
                srgb_img[row][column][channel] = ((323 * cur_val / 25) if cur_val <= RGB_BORDER else \
                                                      ((211 * (cur_val ** (5 / 12)) - 11) / 200))


   # srgb_img = (srgb_img * 255).astype(np.uint8)
    return srgb_img