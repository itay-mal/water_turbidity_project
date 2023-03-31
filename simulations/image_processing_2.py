import cv2
from cv2 import ximgproc
import matplotlib.pyplot as plt
import numpy as np

n_air = 1
n_water = 1.33
focal = 20e-3
target_r = 0.15
sensor_size = 24e-3
num_targets = 2


def main():
    img = cv2.imread("./dataset_for_segmentation/17_gt.png")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    display_image(img_rgb, 'Original Image')
    pre_proc_img = pre_processing(img, True, gamma=1.5)
    circles = cv2.HoughCircles(image=pre_proc_img, method=cv2.HOUGH_GRADIENT, dp=0.5, minDist=100, param2=5)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        target_dist = calc_distance(np.shape(img)[0], circles)
        show_detected_targets(img_rgb, circles)
        print(target_dist)
    plt.show()


def pre_processing(img, show_mid_res=False, gamma=1):
    """
    performs pre-processing pipe consists of contrast-stretch->gamma correction->DOG->thresholding
    ->median filtering
    :param img: input image in BGR
    :param show_mid_res: 'TRUE' to show intermediate results
    :param gamma: value for gamma correction
    :return: pre-processed binary edges image
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_stretch = contrast_stretch(img_gray)
    # img_stretch = cv2.GaussianBlur(img_stretch, (5, 5), 5, sigmaY=5)

    if show_mid_res:
        display_image(img_stretch, "stretch")

    gamma_cor = adjust_gamma(img_stretch, gamma)
    if show_mid_res:
        display_image(gamma_cor, "Gamma Corrected")

    edges = cv2.Canny(gamma_cor, 100, 200)
    if show_mid_res:
        display_image(edges, "Edges")

    # _, edges_thresh = cv2.threshold(edges, 0.5 * np.max(edges), 255, cv2.THRESH_BINARY)
    # if show_mid_res:
    #     display_image(edges_thresh, "Edges Threshold")
    #
    # edges_median = cv2.medianBlur(edges_thresh, 3)
    # if show_mid_res:
    #     display_image(edges_median, "Edges Filtered with Median")

    return edges

def show_hist_rgb(img_rgb):
    """
    plot histograms for color channels
    :param img_rgb:
    :return: none
    """
    fig = plt.figure(figsize=(8, 5))
    axr = fig.add_subplot(1, 3, 1)
    axr.hist(img_rgb[:, :, 0].flatten(), bins=256, range=[0, 256])
    axr.set_title('red channel histogram')
    axg = fig.add_subplot(1, 3, 2)
    axg.hist(img_rgb[:, :, 1].flatten(), bins=256, range=[0, 256])
    axg.set_title('green channel histogram')
    axb = fig.add_subplot(1, 3, 3)
    axb.hist(img_rgb[:, :, 2].flatten(), bins=256, range=[0, 256])
    axb.set_title('blue channel histogram')


def adjust_gamma(image, gamma):
    """
    gamma correction for image
    :param image: image in grayscale [0:255]
    :param gamma: value for gamma correction
    :return: gamma corrected image in grayscale [0:255]
    """
    # build a lookup table mapping the pixel values [0, 255] to
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def contrast_stretch(image):
    """
    :param image: image in grayscale [0:255]
    :return: contrast stretched image in grayscale [0:255]
    """
    image_norm = image / 255
    image_stretch = (image_norm - np.min(image_norm)) / (np.max(image_norm) - np.min(image_norm))
    return (image_stretch * 255).astype(np.uint8)


def diff_gaussian(image, sigma1, sigma2, k_size=3):
    """
    calculates difference of gaussian filtered images as approximation for laplacian of gaussian
    for edge detection
    :param image: image in grayscale [0:255]
    :param sigma1: sigma for first gaussian filter
    :param sigma2: sigma for second gaussian filter
    :param k_size: gaussian filter kernel size
    :return: edges image in grayscale [0:255]
    """
    blur1 = cv2.GaussianBlur(image, (k_size, k_size), sigma1, sigmaY=sigma1).astype(np.int64)
    blur2 = cv2.GaussianBlur(image, (k_size, k_size), sigma2, sigmaY=sigma2).astype(np.int64)
    DOG = np.subtract(blur1, blur2)
    DOG = DOG - np.min(DOG)
    DOG = DOG*255/np.max(DOG)

    return DOG.astype(np.uint8)


def display_image(img, title: str):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img, cmap="gray")
    ax.set_title(title)


def calc_distance(img_height, circles):
    """
    calculate the distance from the camera to the targets in meters
    :param img_height: in pixels
    :param circles: containing for each target data in format (target_center_x,target_center_y, radius)
    in pixels
    :return: for each target return distances in format
    (target_center_x[pixel],target_center_y[pixel], distance[m])
    """
    distances = []
    focal_eff = focal * (n_water / n_air)
    pix_d = sensor_size / img_height
    for x, y, r in circles[:num_targets]:
        est_distance = (focal_eff * target_r) / (pix_d * r)
        distances.append((x, y, est_distance))
    return distances


def show_detected_targets(img_rgb, circles):
    """
    plot green circles around the detected targets
    :param img_rgb:
    :param circles: containing for each target data in format (target_center_x,target_center_y, radius)
    in pixels
    :return: none
    """
    marked = img_rgb.copy()
    for (x, y, r) in circles[:num_targets]:
        marked = cv2.circle(marked, (x, y), r, (0, 255, 0), 1)
    display_image(marked, "Detected Targets")


if __name__ == "__main__":
    main()
