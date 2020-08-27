
import numpy as np

def histogram_match(input_source_img, input_template_img):

    source_shape = input_source_img.shape
    input_source_img = input_source_img.ravel()
    input_template_img = input_template_img.ravel()

    source_values, bin_idx, source_counts = np.unique(input_source_img, return_inverse=True,
                                            return_counts=True)
    template_values, template_counts = np.unique(input_template_img, return_counts=True)

    source_quantiles = np.cumsum(source_counts).astype(np.float64)
    source_quantiles /= source_quantiles[-1]
    template_quantiles = np.cumsum(template_counts).astype(np.float64)
    template_quantiles /= template_quantiles[-1]

    interp_template_values = np.interp(source_quantiles, template_quantiles, template_values)

    return interp_template_values[bin_idx].reshape(source_shape)

def raw_mean_combination(input_imgs):
    imgs = np.asarray(input_imgs)
    output_img = np.mean(imgs, axis=0)
    return output_img

def calibrated_mean_combination(input_source_img, input_template_img):
    calibrated_source_img = histogram_match(input_source_img, input_template_img)
    output_img = np.add(input_template_img, calibrated_source_img) / 2.0
    return output_img

def weighted_calibrated_combination(input_source_img, input_template_img, alpha = 0.1):
    calibrated_source_img = histogram_match(input_source_img, input_template_img)
    output_img = np.add(np.add(input_template_img, (1.0 - alpha) * calibrated_source_img), alpha * input_source_img) / 2.0
    return output_img

def first_stage_integration(input_imgs):

    integral_imgs = []

    for i in range(1, len(input_imgs) + 1):
        integral_imgs.append(raw_mean_combination(input_imgs[:i]))

    result_img = integral_imgs[len(input_imgs) - 1]

    return (integral_imgs, result_img)

def second_stage_integration(input_imgs):

    integral_imgs = []
    result_img = input_imgs[0]

    integral_imgs.append(result_img)
    for i in range(1, len(input_imgs)):
        result_img = calibrated_mean_combination(input_imgs[i], result_img)
        integral_imgs.append(result_img)

    return (integral_imgs, result_img)

def third_stage_integration(input_imgs):

    integral_imgs = []
    result_img = input_imgs[0]

    integral_imgs.append(result_img)
    for i in range(1, len(input_imgs)):
        result_img = weighted_calibrated_combination(input_imgs[i], result_img, alpha = 0.1)
        integral_imgs.append(result_img)

    return (integral_imgs, result_img)

dict_integration_func = {"first_stage": first_stage_integration, "second_stage": second_stage_integration, \
                          "third_stage": third_stage_integration}