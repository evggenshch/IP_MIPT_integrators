import numpy as np
import copy


SRGB_BORDER = 0.04045
RGB_BORDER = 0.0031308

#   Define custom gamma correction


def from_srgb_to_rgb(srgb_img):

    rgb_img = (copy.deepcopy(srgb_img))

    for row, cur_row in enumerate(rgb_img):
        for column, cur_column in enumerate(cur_row):
            for channel, cur_channel in enumerate(cur_column):
                cur_val = np.float32(srgb_img[row][column][channel])
                rgb_img[row][column][channel] =  ((25 * cur_val / 323) if cur_val <= SRGB_BORDER else
                                                      ((200 * cur_val + 11) / 211) ** (12 / 5))

    return rgb_img

def from_rgb_to_srgb(rgb_img):

    srgb_img = copy.deepcopy(rgb_img)

    for row, cur_row in enumerate(srgb_img):
        for column, cur_column in enumerate(cur_row):
            for channel, cur_channel in enumerate(cur_column):
                cur_val = np.float32(rgb_img[row][column][channel])
                srgb_img[row][column][channel] = ((323 * cur_val / 25) if cur_val <= RGB_BORDER else
                                                      ((211 * (cur_val ** (5 / 12)) - 11) / 200))

    return srgb_img



def image_brightness_prealignment(input_img, mode="Shoes-sRGB.jpg"):

    if mode == "Shoes-sRGB.jpg":
        output_img = input_img * 0.9
    else:
        output_img = input_img

    return output_img