import cv2
import numpy as np
import copy
from skimage import color
from skimage import io
import random
import math



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





class VideoFlow(object):

    def __init__(self, image_path):
        self.image = io.imread(image_path)
        self.max_brightness_variation = 1000
        if(image_path == "Shoes-sRGB.jpg"):    #   TO-DO:  change for general case


            self.max_brightness_variation = 1.1

#    def __getitem__(self, key):
#        return self.frames[key]





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