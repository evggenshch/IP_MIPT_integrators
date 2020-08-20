import cv2
import numpy as np
import copy
from skimage import color
from skimage import io
import random
import math

SRGB_GAMMA = 2.2

#   Define custom gamma correction

def from_srgb_to_rgb(srgb_img):
    rgb_img = ((srgb_img / 255) ** SRGB_GAMMA) * 255
    return rgb_img

def from_rgb_to_srgb(rgb_img):
    srgb_img = ((rgb_img / 255) ** (1 / SRGB_GAMMA)) * 255
    return srgb_img

#   Define variation functions

##############
#   Deprecated

#def simple_function(value, y, x, variation_delta_random = 0.005, variation_delta_fixed = 0.0001):
#    random_delta_1 = variation_delta_random * (random.random() - 0.5)
#    random_delta_2 = variation_delta_random * (random.random() - 0.5)
#    coef_1 = ((1.0 + random_delta_1 + variation_delta_fixed) if (random_delta_1 > 0) else (1.0 + random_delta_1 - variation_delta_fixed))
#    coef_2 = ((1.0 + random_delta_2 + variation_delta_fixed) if (random_delta_2 > 0) else (1.0 + random_delta_2 - variation_delta_fixed))
#    return (value * (1.0 + coef_1 * np.log(np.float(y) + 2)) * (1.0 + coef_2 * np.log(np.float(x) + 2)))

##############





def brightness_variation(img, variation_delta_fixed = 0.05, variation_delta_random = 0.1, \
                             variation_func = None):

    augmented_img = copy.deepcopy(img)
    transformed_img = from_srgb_to_rgb(augmented_img)

    if(variation_func == None):
        random_delta = variation_delta_random * (random.random() - 0.5)
        transformed_img = (transformed_img * ((1.0 + random_delta + variation_delta_fixed) if (random_delta > 0) \
                                                                    else (1.0 + random_delta - variation_delta_fixed)))
    else:
        random_delta = variation_delta_random * (random.random() - 0.5)
        transformed_img = (transformed_img * ((1.0 + random_delta + variation_delta_fixed) if (random_delta > 0) \
                                                                    else (1.0 + random_delta - variation_delta_fixed)))
        for row, cur_row in enumerate(transformed_img):
            for column, cur_column in enumerate(cur_row):
                for channel, cur_channel in enumerate(cur_column):
                    transformed_img[row][column][channel] = variation_func(transformed_img[row][column][channel], \
                                                                             row, column)

    augmented_img = from_rgb_to_srgb(transformed_img) (color.xyz2rgb(transformed_img) * 255).astype(dtype = np.uint8)
    return augmented_img

def color_variation(img, variation_delta_fixed = 0.01, variation_delta_random = 0.1):
    augmented_img = copy.deepcopy(img)
    transformed_img = color.rgb2xyz(augmented_img)

    random_delta = variation_delta_random * (random.random() - 0.5)
    transformed_img[:, :, 0] = transformed_img[:, :, 0] * ((1.0 + random_delta + variation_delta_fixed) if (random_delta > 0) \
                                                                    else (random_delta - variation_delta_fixed))
    random_delta = variation_delta_random * random.random()
    transformed_img[:, :, 1] = transformed_img[:, :, 1] * ((1.0 + random_delta + variation_delta_fixed) if (random_delta > 0) \
                                                                    else (random_delta - variation_delta_fixed))
    random_delta = variation_delta_random * random.random()
    transformed_img[:, :, 2] = transformed_img[:, :, 2] * ((1.0 + random_delta + variation_delta_fixed) if (random_delta > 0) \
                                                                    else (random_delta - variation_delta_fixed))

    augmented_img = (color.xyz2rgb(transformed_img) * 255).astype(dtype=np.uint8)
    return augmented_img


class VideoFlow(object):

    def __init__(self, image_path):
        self.image = io.imread(image_path)
        self.first_stage_frames = []
        self.second_stage_frames = []
        self.third_stage_frames = []

#    def __getitem__(self, key):
#        return self.frames[key]



    def first_stage_augmentation(self, num_frames = 100):
        ref_frame = self.image
        for i in range(num_frames):
            cur_frame = brightness_variation(ref_frame, variation_delta_fixed= 0.01, variation_delta_random = 0.1)
            self.first_stage_frames.append(copy.deepcopy(cur_frame))

    def second_stage_augmentation(self, num_frames = 100):
        ref_frame = self.image
        for i in range(num_frames):
            cur_frame = brightness_variation(ref_frame, variation_func=simple_function)
            self.second_stage_frames.append(copy.deepcopy(cur_frame))

    def third_stage_augmentation(self, num_frames = 100):
        ref_frame = self.image
        for i in range(num_frames):
            cur_frame = color_variation(ref_frame)
            cur_frame = brightness_variation(cur_frame, variation_func=simple_function)
            self.third_stage_frames.append(copy.deepcopy(cur_frame))