import numpy as np
import copy
from skimage import color
from skimage import io
import random
import math


SRGB_BORDER = 0.04045
RGB_BORDER = 0.0031308

#   Define custom gamma correction


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



def image_brightness_alignment(input_img, mode="Shoes-sRGB.jpg"):

    if mode == "Shoes-sRGB.jpg":
        output_img = (from_rgb_to_srgb(from_srgb_to_rgb(input_img).astype(dtype=np.float32) / 255 * 0.9) \
                  * 255).astype(dtype=np.uint8)

    return output_img