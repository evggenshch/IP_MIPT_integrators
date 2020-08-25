
import numpy as np
import math
import integration
import luminosity_videoflow
from augmentation import first_stage_augmentation, second_stage_augmentation, third_stage_augmentation, dict_augmentation_func
from integration import  first_stage_integration, second_stage_integration, third_stage_integration, dict_integration_func
from utils import l2_metric




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

    convergence_rate = np.log10(np.abs((imgs[N - 1] - imgs[N - 2]) / (imgs[N - 2] - imgs[N - 3]))) \
                       / np.log10(np.abs((imgs[N - 2] - imgs[N - 3]) / (imgs[N - 3] - imgs[N - 4])))

    return (convergence_rate, convergence_flag)


def conduct_experiment(videoflow, stage_augmentation = "first_stage", stage_integration = "first_stage", num = 1000):

    converged_num = 0
    convergence_rates = np.array()

    for cur_experiment in range(num):
        np.random.seed(42)
        convergence_rate, convergence_flag = measure_convergence()
        if convergence_flag:
            converged_num += 1
            convergence_rates.append(convergence_rate)
        pass

    convergence_rates = np.array(convergence_rates)

    return (convergence_rates, converged_num)
