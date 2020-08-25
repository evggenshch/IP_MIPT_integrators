
import numpy as np
import copy
import random
from skimage import color
from utils import from_srgb_to_rgb, from_rgb_to_srgb

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

def first_stage_augmentation(input_img, random_seed):

    output_img = brightness_variation(input_img, variation_delta_fixed= 0.01, variation_delta_random = 0.1)

   # def first_stage_augmentation(self, num_frames = 10):
       # ref_frame = input_img
       # for i in range(num_frames):
       #     self.first_stage_frames.append(copy.deepcopy(cur_frame))

    return output_img

def second_stage_augmentation(input_img, random_seed):

    output_img = brightness_variation(input_img, variation_delta_fixed= 0.01, variation_delta_random = 0.1)
    output_img = color_variation(output_img, variation_delta_fixed = 0.01, variation_delta_random = 0.1)

    return output_img

def third_stage_augmentation(input_img, random_seed):

    output_img = brightness_variation(input_img, variation_delta_fixed= 0.01, variation_delta_random = 0.1)
    output_img = color_variation(output_img, variation_delta_fixed = 0.01, variation_delta_random = 0.1)

    return output_img

dict_augmentation_func = {"first_stage": first_stage_augmentation, "second_stage": second_stage_augmentation, \
                          "third_stage": third_stage_augmentation}