from image_processing_AC import AC_detction
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from matplotlib.patches import Ellipse
from matplotlib.transforms import Bbox
from skimage import io
from skimage.color import rgb2gray
from myEllipseRansac import myEllipseRansac
from drag_ellipse import DraggablePlot
import easygui

from tqdm import tqdm
import time


N_AIR = 1
N_WATER = 1 # 1.33
FOCAL = 20e-3
TARGET_R = 0.15
SENSOR_SIZE = 24e-3 # for 35mm sensor: 24X36mm

path = 'C:/Users/itaym/Desktop/000.png' #"C:/Users/nitay/Desktop/0000.png"

def main():
    print('start: {}'.format(time.time()))
    img = io.imread(path)
    img_gray = rgb2gray(img)
    print('before AC detection: {}'.format(time.time()))
    # get snakes 
    snake_1, snake_2 = AC_detction(img, show_intermediate_results=True)
    print('after AC detection: {}'.format(time.time()))

    # get estimated ellipses from snakes
    target_1 = myEllipseRansac(snake_1).get_params()
    target_2 = myEllipseRansac(snake_2).get_params()

    # user correct estimated ellipses
    DraggablePlot(image=img,
                  target1_est=target_1,
                  target2_est=target_2,
                  my_callback=calc_coeffs_from_ellipses
                  )
    
def calc_coeffs_from_ellipses(target1, target2, img):
    """
    calculate the attenuation coeffs given the marked targets
    called as callback from DraggablePlot, 
    results are displayed in pop-up window
    inputs:
        target1/2 - ellipses params ((x,y),width,height,theta)
        image     - the original rgb image
    """
    # make targets into matplotlib patches so we can use some nice features
    t1_patch = Ellipse(*target1)
    t2_patch = Ellipse(*target2)

    t1_TL, t1_BR = t1_patch.get_extents().get_points()
    t2_TL, t2_BR = t2_patch.get_extents().get_points()

    targets_TL = np.min(np.vstack((t1_TL, t2_TL)), axis=0).astype(int)
    targets_BR = np.max(np.vstack((t1_BR, t2_BR)), axis=0).astype(int)

    img_gray = rgb2gray(img)
    mask_1 = np.ndarray(img.shape[:2], bool)
    mask_2 = np.ndarray(img.shape[:2], bool)
    t1 = []
    t2 = []

    # TODO: do we still want to do the for loop?
    print('before for loop: {}'.format(time.time()))
    for i in tqdm(range(targets_TL[1], targets_BR[1] + 1), position=0): # row
        for j in tqdm(range(targets_TL[0], targets_BR[0] + 1), position=1, leave=False): # column
            if t1_patch.contains_point((j, i)):
                mask_1[i, j] = True
                t1.append((i, j, img_gray[i,j], *list(img[i,j])))  # (y,x,gray,r,g,b)
            if t2_patch.contains_point((j, i)):
                mask_2[i, j] = True
                t2.append((i, j, img_gray[i,j], *list(img[i, j])))  # (y,x,gray,r,g,b)
    print('after for loop: {}'.format(time.time()))
    t1 = np.array(t1)
    t2 = np.array(t2)

    # TODO: now we are dealing with user marked ellipse, is this relevant?
    # center_t1, r1 = get_center_radius_from_snake(np.array(target_1), t1)
    # center_t2, r2 = get_center_radius_from_snake(np.array(target_2), t2)

    # t1 = t1[np.linalg.norm(t1[:,:2] - center_t1, axis=1) < 0.9*r1]
    # t2 = t2[np.linalg.norm(t2[:,:2] - center_t2, axis=1) < 0.9*r2]

    # TODO: we calculate distance with radius, how to define in ellipse? 
    # height? width?
    r1 = target1[2] / 2
    r2 = target2[2] / 2

    d1 = calc_distance(img.shape[0], r1)
    d2 = calc_distance(img.shape[0], r2)

    avg_t1 = np.mean(t1[:, 2])
    avg_t2 = np.mean(t2[:, 2])
    t1_b = t1[t1[:, 2] < avg_t1]
    t1_w = t1[t1[:, 2] >= avg_t1]
    t2_b = t2[t2[:, 2] < avg_t2]
    t2_w = t2[t2[:, 2] >= avg_t2]

    mask_3 = np.ndarray(img.shape, int)
    mask_3[t1_w[:, 0].astype(int), t1_w[:, 1].astype(int)] = (0,0,255)
    mask_3[t1_b[:, 0].astype(int), t1_b[:, 1].astype(int)] = (255,0,0)
    mask_3[t2_w[:, 0].astype(int), t2_w[:, 1].astype(int)] = (0,255,0)
    mask_3[t2_b[:, 0].astype(int), t2_b[:, 1].astype(int)] = (255,255,255)

    avg_t1_b = np.mean(t1_b[:, 3:], axis=0)
    avg_t1_w = np.mean(t1_w[:, 3:], axis=0)
    avg_t2_b = np.mean(t2_b[:, 3:], axis=0)
    avg_t2_w = np.mean(t2_w[:, 3:], axis=0)


    att_R_w, att_G_w, att_B_w = clac_attenuation_coeffs(d1, avg_t1_w, avg_t1_b, d2, avg_t2_w, avg_t2_b)

    print(att_R_w, att_G_w, att_B_w, d1, d2)
    easygui.msgbox(f"Attenuation Coeffs:\n"
                   f"R: {att_R_w}\tG: {att_G_w}\tB: {att_B_w}"
                   f"\n\nDists:\nTarget1: {d1}\tTarget2: {d2}", 
                   title="Results")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(mask_3)
    plt.show()

def clac_attenuation_coeffs(t1_dist, t1_w_avg, t1_b_avg, t2_dist, t2_w_avg, t2_b_avg):
    """
    calculate attenuation coeeficients for RGB chnnel, expects the image to be in RGB
    :return: attenuation coeffs in RGB format
    """
    att_R_w = - np.log((t1_w_avg[0]-t1_b_avg[0])/(t2_w_avg[0]-t2_b_avg[0])) / (t1_dist - t2_dist)
    att_G_w = - np.log((t1_w_avg[1]-t1_b_avg[1])/(t2_w_avg[1]-t2_b_avg[1])) / (t1_dist - t2_dist)
    att_B_w = - np.log((t1_w_avg[2]-t1_b_avg[2])/(t2_w_avg[2]-t2_b_avg[2])) / (t1_dist - t2_dist)
    return att_R_w, att_G_w, att_B_w


def calc_distance(img_height, radius):
    """
    calculate the distance from the camera to the targets in meters
    :param img_height: in pixels
    :param radius: radius of the target in pixels
    :return: for each target return distance in meters
    """
    focal_eff = FOCAL * (N_WATER / N_AIR)
    pix_d = SENSOR_SIZE / img_height
    return (focal_eff * TARGET_R) / (pix_d * radius)

def get_center_radius_from_snake(snake,t):
    """
    calculate the radius and center of a AC detection in pixels.
    expect snake to be np.array(n,2) and t are the pixels inside each target in format [n,(y,x,gray,r,g,b)].
    """
    c = np.mean(t[:,:2], axis=0)
    r = np.mean(np.linalg.norm(np.array(snake) - c, axis=1))

    return c, r

if __name__ == "__main__":
    main()
