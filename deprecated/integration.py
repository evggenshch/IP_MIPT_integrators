
from skimage import color
from skimage import io
import numpy as np

def first_phase_integrator(images):
    imgs = np.asarray(images)
    metrics = []

    for i in range(len(imgs) - 1):
        next = imgs[i + 1]
        metrics.append(np.linalg.norm(next - np.mean(imgs[:i + 1], axis=0)))

    average_img = np.mean(imgs, axis=0).astype(dtype= np.uint8)
    return (metrics, average_img)

def second_phase_integrator(images):
    imgs = np.asarray(images)
    metrics = []

    for i in range(1, len(imgs)):
        metrics.append()
        pass

    # Take the median over the first dim
    #print(imgs)
    #print(imgs.shape)
    average_img = np.mean(imgs, axis=0)
    return (metrics, average_img)

def third_phase_integrator(images):
    imgs = np.asarray(images)
    metrics = []

    for i in range(len(imgs)):
        pass

    # Take the median over the first dim
    #print(imgs)
    #print(imgs.shape)
    average_img = np.mean(imgs, axis=0)
    return (metrics, average_img)