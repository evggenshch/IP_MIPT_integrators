
import numpy as np
import copy

def sine_variation(x):
    return np.sin(x)

def cosine_variation(x):
    return np.cos(x)

def neg_exp_variation(x):
    return np.exp(- x ** 2)

def default_brightness_variation_func(input_pixel, row, column):
    return np.clip(0.2, 0.9, np.sqrt(np.square(sine_variation(0.3 * row) * input_pixel)
                                     + np.square(cosine_variation(0.1 * column) * input_pixel)))

def brightness_variation(input_img, random_seed = 42, variation_delta_fixed = 0.05, variation_delta_random = 0.1,
                             variation_func = None):

    np.random.seed(random_seed)

    if (variation_func == None):
        random_delta = variation_delta_random * (np.random.random() - 0.5)
        augmented_img = (input_img * ((1.0 + random_delta + variation_delta_fixed) if (random_delta > 0)
                                                                    else (1.0 + random_delta - variation_delta_fixed)))
    else:
        random_delta = variation_delta_random * (np.random.random() - 0.5)
        augmented_img = (input_img * ((1.0 + random_delta + variation_delta_fixed) if (random_delta > 0)
                                      else (1.0 + random_delta - variation_delta_fixed)))

        dim_augmet_img = lambda t, row, column: variation_func(t, row, column)
        vectorized_dim_augmet_img = np.vectorize(dim_augmet_img)

        augmented_img = vectorized_dim_augmet_img(augmented_img, np.repeat((np.arange(0, augmented_img.shape[0], None)), augmented_img.shape[1], axis=1),
                                                 np.repeat(np.arange(0, augmented_img.shape[1]).transpose(), augmented_img.shape[0], axis=0))

   #     for row, cur_row in enumerate(input_img):
   #         for column, cur_column in enumerate(cur_row):
   #             for channel, cur_channel in enumerate(cur_column):
   #                 augmented_img[row][column][channel] = variation_func(input_img[row][column][channel],
   #                                                                          row, column)

    return augmented_img

def color_variation(input_img, random_seed = 42, variation_delta_fixed = 0.01, variation_delta_random = 0.1,       # +.
                    variation_func = None):

    np.random.seed(random_seed)
    augmented_img = input_img

    if (variation_func == None):

        random_delta = variation_delta_random * (np.random.random() - 0.5)
        augmented_img[:, :, 0] = input_img[:, :, 0] * ((1.0 + random_delta + variation_delta_fixed) if (random_delta > 0)
                                                                    else (random_delta - variation_delta_fixed))
        random_delta = variation_delta_random * np.random.random()
        augmented_img[:, :, 1] = input_img[:, :, 1] * ((1.0 + random_delta + variation_delta_fixed) if (random_delta > 0)
                                                                    else (random_delta - variation_delta_fixed))
        random_delta = variation_delta_random * np.random.random()
        augmented_img[:, :, 2] = input_img[:, :, 2] * ((1.0 + random_delta + variation_delta_fixed) if (random_delta > 0)
                                                                    else (random_delta - variation_delta_fixed))

    return augmented_img


def first_stage_augmentation(input_imgs, random_seed = 42, variation_delta_fixed= 0.05, variation_delta_random = 0.1):

    output_imgs = []

    for i, cur_input_img in enumerate(input_imgs):
        output_imgs.append(brightness_variation(cur_input_img, i, variation_delta_fixed, variation_delta_random))

    return output_imgs


def second_stage_augmentation(input_imgs, random_seed = 42, variation_delta_fixed= 0.5, variation_delta_random = 0.5):

    output_imgs = []

    for i, cur_input_img in enumerate(input_imgs):
        output_imgs.append(brightness_variation(cur_input_img, i, variation_delta_fixed, variation_delta_random,
                                          default_brightness_variation_func))

    return output_imgs


def third_stage_augmentation(input_imgs, random_seed = 42, variation_delta_fixed= 0.5, variation_delta_random = 0.5):

    output_imgs = []

    for i, cur_input_img in enumerate(input_imgs):
        cur_output_img = brightness_variation(cur_input_img, i, variation_delta_fixed, variation_delta_random)
        output_imgs.append(color_variation(cur_output_img, i, variation_delta_fixed, variation_delta_random))

    return output_imgs


dict_augmentation_func = {"first_stage_aug": first_stage_augmentation, "second_stage_aug": second_stage_augmentation,
                          "third_stage_aug": third_stage_augmentation}