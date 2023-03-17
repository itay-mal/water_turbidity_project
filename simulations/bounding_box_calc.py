import cv2
import matplotlib.pyplot as plt
import numpy as np

n_air = 1
n_water = 1.33
focal = 20e-3
target_r = 0.15
sensor_size = 24e-3
num_targets = 2

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