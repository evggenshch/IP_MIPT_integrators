
import numpy as np
from .augmentation import dict_augmentation_func
from .integration import  dict_integration_func
from sklearn.metrics import mean_squared_error
from .utils import from_srgb_to_rgb, from_rgb_to_srgb, image_brightness_prealignment
from skimage import io
import os
import shutil
import copy
from tqdm import tqdm

def l2_metric(img_prev, img_next):

    L2 = np.sum((img_prev.astype(np.float) - img_next.astype(np.float)) ** 2)
    L2 /= np.float(img_prev.shape[0] * img_prev.shape[1])

    return L2


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
    input_img = np.array(input_img)
    transformed_input_img = from_srgb_to_rgb(copy.deepcopy(input_img).astype(dtype = np.float32) / 255.0)
    transformed_input_img = image_brightness_prealignment(transformed_input_img)
    raw_imgs = videoflow_size * [transformed_input_img]

    if os.path.exists(integrated_imgs_save_base_dir):
        shutil.rmtree(integrated_imgs_save_base_dir)
    os.makedirs(integrated_imgs_save_base_dir)

    for cur_experiment in tqdm(range(num_experiments)):

        print("EXPERIMENT #{} CONDUCTING...".format(cur_experiment))

        cur_experiment_save_dir = str(integrated_imgs_save_base_dir) + "/experiment_" \
                                  + str(cur_experiment).zfill(int(np.log10(num_experiments)))

        cur_experiment_raw_dir = str(integrated_imgs_save_base_dir) + "/raw_" \
                                  + str(cur_experiment).zfill(int(np.log10(num_experiments)))

        augmented_imgs = dict_augmentation_func[stage_augmentation](raw_imgs)

        if os.path.exists(cur_experiment_raw_dir):
            shutil.rmtree(cur_experiment_raw_dir)
        os.makedirs(cur_experiment_raw_dir)

        for img_num, cur_augmented_img in tqdm(enumerate(augmented_imgs)):
            io.imsave(cur_experiment_raw_dir + "/img_" + str(img_num).zfill(int(np.log10(videoflow_size))) + ".png",
                      (from_rgb_to_srgb(cur_augmented_img) * 255).astype(dtype = np.uint8))

        integrated_imgs, result_img = dict_integration_func[stage_integration](augmented_imgs)

        #print(len(integrated_imgs))

        convergence_rate, convergence_flag = measure_convergence(integrated_imgs, convergence_interval=0.1)
        if convergence_flag:
            converged_num += 1
            convergence_rates.append(convergence_rate)

        if os.path.exists(cur_experiment_save_dir):
            shutil.rmtree(cur_experiment_save_dir)
        os.makedirs(cur_experiment_save_dir)

        print(np.max(((from_rgb_to_srgb(integrated_imgs[len(integrated_imgs) - 1]) * 255).astype(dtype=np.uint8))))

        for img_num, cur_integrated_img in tqdm(enumerate(integrated_imgs)):
            io.imsave(cur_experiment_save_dir + "/img_" + str(img_num).zfill(int(np.log10(videoflow_size))) + ".png",
                      (from_rgb_to_srgb(cur_integrated_img) * 255).astype(dtype = np.uint8))

    #    io.imsave(cur_experiment_save_dir + "/result_img",
    #              (from_rgb_to_srgb(integrated_imgs[len(integrated_imgs) - 1]) * 255).astype(dtype = np.uint8))

        io.imsave(cur_experiment_save_dir + "/result_img.png",
                  (from_rgb_to_srgb(integrated_imgs[len(integrated_imgs) - 1]) * 255).astype(dtype = np.uint8))

    convergence_rates = np.array(convergence_rates)

    return (convergence_rates, converged_num)
