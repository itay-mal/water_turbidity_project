import cv2
import skimage
from skimage.feature import canny
import sklearn
from scipy import ndimage as ndi

import matplotlib.pyplot as plt
import numpy as np

n_air = 1
n_water = 1 #1.33
focal = 20e-3
target_r = 0.15
sensor_size = 24e-3
num_targets = 2


def main():
    img = cv2.imread("./attenuation_coeffs_sweep/000.png")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    display_image(img_rgb, 'Original Image')
    circles = detect_circles(img_rgb)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        target_dist = calc_distance(np.shape(img)[0], circles)
        show_detected_targets(img_rgb, circles)
        print(target_dist)
    plt.show()


def detect_circles(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    display_image(gray, 'grayscale')
    red_chan = img_rgb[:, :, 0]
    display_image(red_chan, 'red')
    grn_chan = img_rgb[:, :, 1]
    display_image(grn_chan, 'green')
    blu_chan = img_rgb[:, :, 2]
    display_image(blu_chan, 'blue')
    edges = canny(gray/255.)
    # filled_targets = (255 * ndi.binary_fill_holes(edges).astype(np.uint8))
    # display_image(filled_targets, 'filled targets')
    kernel = np.ones((3, 3), np.uint8)
    edges_dialated = cv2.dilate(255*edges.astype(np.uint8), kernel, iterations=1)
    display_image(edges_dialated, 'edges_dialated')

    return cv2.HoughCircles(image=edges_dialated, method=cv2.HOUGH_GRADIENT, dp=0.5, minDist=50, param2=5)


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
