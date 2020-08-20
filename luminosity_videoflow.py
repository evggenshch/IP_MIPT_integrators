import cv2
import numpy as np
import copy
from skimage import color
from skimage import io
import random
import math

SRGB_BORDER = 0.04045
RGB_BORDER = 0.0031308

#SRGB_RGB_MATRIX = np.array([[0.18048079, 0.07219232, 0.95053215], \
#                            [0.35758434, 0.71516868, 0.11919478], \
#                            [0.41239080, 0.21263901, 0.01933082]])
#RGB_SRGB_MATRIX = np.array([[], [], []])

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

#def threshold_()

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

def sine_variation(x):
    return math.sin(x)

def cosine_variation(x):
    return math.cos(x)

def neg_exp_variation(x):
    return math.exp(- x ** 2)


def brightness_variation(img, variation_delta_fixed = 0.05, variation_delta_random = 0.1, \
                             variation_func = None, max_brightness_variation = 1000):

    augmented_img = copy.deepcopy(img).astype(dtype = np.float32) / 255
    transformed_img = from_srgb_to_rgb(augmented_img)

    if(variation_func == None):
        random_delta = variation_delta_random * (random.random() - 0.5)
        transformed_img = (transformed_img * ((1.0 + random_delta + variation_delta_fixed) if (random_delta > 0) \
                                                                    else (1.0 + random_delta - variation_delta_fixed)))
  #  else:
  #      random_delta = variation_delta_random * (random.random() - 0.5)
  #      transformed_img = (transformed_img * ((1.0 + random_delta + variation_delta_fixed) if (random_delta > 0) \
  #                                                                  else (1.0 + random_delta - variation_delta_fixed)))
  #      for row, cur_row in enumerate(transformed_img):
  #          for column, cur_column in enumerate(cur_row):
  #              for channel, cur_channel in enumerate(cur_column):
  #                  transformed_img[row][column][channel] = variation_func(transformed_img[row][column][channel], \
  #                                                                           row, column)



  #   transformed_img *= 0.8

    augmented_img = (from_rgb_to_srgb(transformed_img) * 255).astype(dtype = np.uint8) #(color.xyz2rgb(transformed_img) * 255).astype(dtype = np.uint8)

   # return transformed_img()
    return augmented_img # ((img.astype(dtype = np.float32) / 255 * 0.8) * 255).astype(dtype = np.uint8) #augmented_img

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
        self.max_brightness_variation = 1000
        if(image_path == "Shoes-sRGB.jpg"):    #   TO-DO:  change for general case

            self.image = (from_rgb_to_srgb(from_srgb_to_rgb(self.image).astype(dtype = np.float32) / 255 * 0.9) \
                          * 255).astype(dtype=np.uint8)
            self.max_brightness_variation = 1.1

        self.first_stage_frames = []
        self.second_stage_frames = []
        self.third_stage_frames = []

#    def __getitem__(self, key):
#        return self.frames[key]



    def first_stage_augmentation(self, num_frames = 10):
        ref_frame = self.image
        for i in range(num_frames):
            cur_frame = brightness_variation(ref_frame, variation_delta_fixed= 0.01, variation_delta_random = 0.1)
            self.first_stage_frames.append(copy.deepcopy(cur_frame))

   # def second_stage_augmentation(self, num_frames = 100):
   #     ref_frame = self.image
   #     for i in range(num_frames):
   #         cur_frame = brightness_variation(ref_frame, variation_func=simple_function)
   #         self.second_stage_frames.append(copy.deepcopy(cur_frame))

   # def third_stage_augmentation(self, num_frames = 100):
   #     ref_frame = self.image
   #     for i in range(num_frames):
   #         cur_frame = color_variation(ref_frame)
   #         cur_frame = brightness_variation(cur_frame, variation_func=simple_function)
   #         self.third_stage_frames.append(copy.deepcopy(cur_frame))