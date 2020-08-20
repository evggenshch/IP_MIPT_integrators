
import numpy as np
import math
import integration
import luminosity_videoflow


def l2_metric(img_prev, img_next):

    return np.linalg.norm(img_next - img_prev)


def measure_convergence(imgs, convergence_interval = 0.1):

    N = len(imgs)
    diffs = []
    convergence_flag = True

    if(isinstance(convergence_interval, float)):
        convergence_interval = int( convergence_interval * len(imgs))

    for i in range(N - 1):
        diffs.append(l2_metric(imgs[i], imgs[i + 1]))
        if (i >= (N - convergence_interval)) and (diffs[i] > diffs[i - 1]):
            convergence_flag = False

    return (diffs, convergence_flag)


def conduct_experiment(videoflow, augmentation, integrator, num = 1000):

    getattr(videoflow, augmentation)()
    getattr()

    for cur_experiment in range(num):
        c        measure_convergence()
        pass

    pass
