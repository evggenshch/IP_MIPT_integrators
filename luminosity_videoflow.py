import cv2
import numpy as np
import copy

class VideoFlow(object):


    def __init__(self, image_path):
        self.image
        pass

    def __getitem__(self, key):
        pass

    def luminosity_augmentation(img):
        augmented_img = copy.deepcopy(img)
        for row, cur_row in enumerate(augmented_img):
            for column, cur_column in enumerate(cur_row):
                for channel, cur_channel in enumerate(cur_column):
                    pass


        return augmented_img

    def first_stage_augmentation(self, num_frames = 100):
        #s
        for i in range(num_frames):
            pass

    def second_stage_augmentation(self, num_frames = 100):
        pass

    def third_stage_augmentation(self, num_frames = 100):
        pass