
import numpy as np
from .augmentation import dict_augmentation_func
from .integration import  dict_integration_func
from sklearn.metrics import mean_squared_error
from .utils import from_srgb_to_rgb, from_rgb_to_srgb, image_brightness_prealignment
from skimage import io
import os
import shutil
import copy

def l2_metric(img_prev, img_next):

    return mean_squared_error(img_next, img_prev)


def measure_convergence(imgs, convergence_interval = 0.1):

    N = len(imgs)
    diffs = []
    convergence_flag = True
    convergence_rate = None

    if(isinstance(convergence_interval, float)):
        convergence_interval = int( convergence_interval * len(imgs))

    for i in range(N - 1):
        diffs.append(l2_metric(imgs[i], imgs[i + 1]))
        if (i >= (N - convergence_interval)) and (diffs[i] > diffs[i - 1]):
            convergence_flag = False
            return(convergence_rate, convergence_flag)

    if convergence_flag:
        convergence_rate = np.log10(np.abs((imgs[N - 1] - imgs[N - 2]) / (imgs[N - 2] - imgs[N - 3]))) \
                       / np.log10(np.abs((imgs[N - 2] - imgs[N - 3]) / (imgs[N - 3] - imgs[N - 4])))

    return (convergence_rate, convergence_flag)


def conduct_experiment(img_path, videoflow_size = 100, stage_augmentation = "first_stage", stage_integration = "first_stage",
                       integrated_imgs_save_base_dir = None, num_experiments = 1000):

    np.random.seed(42)
    converged_num = 0
    convergence_rates = [] #np.array()

    input_img = io.imread(img_path)
    transformed_input_img = from_srgb_to_rgb(copy.deepcopy(input_img).astype(dtype = np.float32) / 255.0)
    transformed_input_img = image_brightness_prealignment(transformed_input_img)
    raw_imgs = videoflow_size * [transformed_input_img]

    if os.path.exists(integrated_imgs_save_base_dir):
        shutil.rmtree(integrated_imgs_save_base_dir)
    os.makedirs(integrated_imgs_save_base_dir)

    for cur_experiment in range(num_experiments):

        cur_experiment_save_dir = str(integrated_imgs_save_base_dir) + "/experiment_" \
                                  + str(cur_experiment).zfill(int(np.log10(num_experiments)))

        augmented_imgs = dict_augmentation_func[stage_augmentation](raw_imgs)
        integrated_imgs = dict_integration_func[stage_integration](augmented_imgs)

        convergence_rate, convergence_flag = measure_convergence(integrated_imgs, convergence_interval=0.1)
        if convergence_flag:
            converged_num += 1
            convergence_rates.append(convergence_rate)

        for img_num, cur_integrated_img in enumerate(integrated_imgs):
            io.imsave(cur_experiment_save_dir + "/img_" + str(img_num).zfill(int(np.log10(videoflow_size))),
                      (from_rgb_to_srgb(cur_integrated_img) * 255).astype(dtype = np.uint8))

        io.imsave(cur_experiment_save_dir + "/result_img",
                  (from_rgb_to_srgb(integrated_imgs[len(integrated_imgs) - 1]) * 255).astype(dtype = np.uint8))

    convergence_rates = np.array(convergence_rates)

    return (convergence_rates, converged_num)
