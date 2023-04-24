import cv2
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import os
from tqdm import tqdm

N_AIR = 1
N_WATER = 1.33
FOCAL = 20e-3
TARGET_R = 0.15
SENSOR_SIZE = 24e-3
num_targets = 2


#### consts from manual GUI ####
T1_X = 240
T1_Y = 250
T1_R = 137
T1_DIST = 0.6
T1_THETA = 0
T2_X = 441
T2_Y = 250
T2_R = 69
T2_DIST = 1.2
T2_THETA = 0
#############################

images_root = "C:/Users/nitay/Desktop/attenuation_coeffs_sweep"
img_path = "C:/Users/nitay/Desktop/000.png"
# img_path = "./white_light_test.png"
THICKNESS = 1


def main():
    coeffs = []

    # get calculation from rendered images
    for f in tqdm(sorted(os.listdir(images_root))):
        if f == "log.txt": continue
        img = cv2.imread(os.path.join(images_root, f))
        coeffs.append([get_att_coeffs(img)])

    # get expected values from log
    with open(os.path.join(images_root, "log.txt"), 'rt') as f:
        for line in f:
            idx, _, _, sigma_s, sigma_a, _ = line.split(':')
            idx = int(idx)
            sigma_a = eval(sigma_a.split('=')[-1])
            sigma_s = eval(sigma_s.split('=')[-1])
            coeffs[idx].append((sigma_a[0] + sigma_s[0], sigma_a[1] + sigma_s[1], sigma_a[2] + sigma_s[2]))

    plot_calculated_vs_expected(coeffs)
    plt.show()

def plot_calculated_vs_expected(coeffs):
    coeffs = np.array(coeffs)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x=coeffs[:,1,0], y=coeffs[:,0,0], s=0.5, c='tab:red', label='red channel')
    ax.scatter(x=coeffs[:,1,1], y=coeffs[:,0,1], s=0.5, c='tab:green', label='green channel')
    ax.scatter(x=coeffs[:,1,2], y=coeffs[:,0,2], s=0.5, c='tab:blue', label='blue channel')
    ax.set_xlim(0,2.5)
    ax.set_ylim(0,2.5)
    ax.set_aspect('equal', 'box')
    ax.legend()
    ax.set_title('attenuation coeffs')
    ax.set_xlabel('calculated')
    ax.set_ylabel('expected')

def get_att_coeffs(img):
    t1_dist = T1_DIST
    t2_dist = T2_DIST
    x1 = T1_X
    y1 = T1_Y
    r1 = T1_R
    x2 = T2_X
    y2 = T2_Y
    r2 = T2_R
    theta1 = T1_THETA  # same for both targets
    theta2 = T1_THETA  # same for both targets
    t1_w = []
    t1_b = []
    t2_w = []
    t2_b = []
    # collect pixels for targets' sections
    for y_hat, x_hat in product(range(img.shape[0]), range(img.shape[1])):
        if in_target(x1, y1, r1, x_hat, y_hat):  # in target 1
            angle = get_angle((x_hat - x1, y1 - y_hat))  # flipped on Y axis bc img coord sys is upside down
            if in_black_sections(theta1, angle):
                t1_b.append(img[y_hat, x_hat])
            else:
                t1_w.append(img[y_hat, x_hat])

        elif in_target(x2, y2, r2, x_hat, y_hat):  # in target 2
            angle = get_angle((x_hat - x2, y2 - y_hat))  # flipped on Y axis bc img coord sys is upside down
            if in_black_sections(theta2, angle):
                t2_b.append(img[y_hat, x_hat])
            else:
                t2_w.append(img[y_hat, x_hat])
    t1_b = np.array(t1_b)
    t1_w = np.array(t1_w)
    t2_b = np.array(t2_b)
    t2_w = np.array(t2_w)
    return clac_attenuation_coeffs(t1_dist, t1_w, t1_b, t2_dist, t2_w, t2_b)


def in_target(x, y, r, x_hat_, y_hat_):
    """check if given pixel (x_hat, y_hat) is inside a target with center coords (x,y) and radius r"""
    return np.linalg.norm(np.array([x, y]) - np.array([x_hat_, y_hat_])) < r

def in_black_sections(theta, angle_):
    """check if the angle is in the correct ranges to be in the black sections of the target"""
    if 0 < theta <= np.pi/2:
        return np.tan(theta) <= np.tan(angle_) or np.tan(angle_) <= np.tan(theta + np.pi / 2)
    return np.tan(theta) <= np.tan(angle_) <= np.tan(theta + np.pi / 2)



def clac_attenuation_coeffs(t1_dist, t1_w, t1_b, t2_dist, t2_w, t2_b):
    """
    calculate attenuation coeeficients for RGB chnnel, expects the image to be in BGR
    :param t1_dist:
    :param t1_w:
    :param t1_b:
    :param t2_dist:
    :param t2_w:
    :param t2_b:
    :return: attenuation coeefs in RGB format
    """
    t1_w_avg = np.average(t1_w, axis=0)
    # t1_b_avg = np.average(t1_b, axis=0)
    t2_w_avg = np.average(t2_w, axis=0)
    # t2_b_avg = np.average(t2_b, axis=0)
    att_B_w = - np.log(t1_w_avg[0]/t2_w_avg[0]) / (t1_dist - t2_dist)
    att_G_w = - np.log(t1_w_avg[1]/t2_w_avg[1]) / (t1_dist - t2_dist)
    att_R_w = - np.log(t1_w_avg[2]/t2_w_avg[2]) / (t1_dist - t2_dist)
    # att_B_b = - np.log(t1_b_avg[0]/t2_b_avg[0]) / (t1_dist - t2_dist)
    # att_G_b = - np.log(t1_b_avg[1]/t2_b_avg[1]) / (t1_dist - t2_dist)
    # att_R_b = - np.log(t1_b_avg[2]/t2_b_avg[2]) / (t1_dist - t2_dist)
    # print(f'attenuation blue on white: {att_B_w}')
    # print(f'attenuation blue on black: {att_B_b}')
    # print(f'attenuation blue average: {np.average([att_B_w,att_B_b])}')
    # print(f'attenuation green on white: {att_G_w}')
    # print(f'attenuation green on black: {att_G_b}')
    # print(f'attenuation green average: {np.average([att_G_w,att_G_b])}')
    # print(f'attenuation red on white: {att_R_w}')
    # print(f'attenuation red on black: {att_R_b}')
    # print(f'attenuation red average: {np.average([att_R_w,att_R_b])}')
    return (att_R_w, att_G_w, att_B_w)



def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def get_angle(v1):
    """ Returns the angle in radians of vector 'v1'"""
    v1_u = unit_vector(v1)
    angle = np.arctan(v1_u[1] / v1_u[0])
    if v1_u[0] < 0:  # 2nd and 3rd quadrants
        angle += np.pi
    elif v1_u[1] < 0:  # 4th quadrant
        angle += 2 * np.pi
    return angle


def draw_target_shape(img, x, y, r, theta, color):
    cv2.circle(img, (x, y), r, color, THICKNESS)
    cv2.line(img,
             (int(x - r * np.cos(theta)), int(y + r * np.sin(theta))),
             (int(x + r * np.cos(theta)), int(y - r * np.sin(theta))),
             color,
             THICKNESS)
    cv2.line(img,
             (int(x - r * np.cos(theta + np.pi / 2)), int(y + r * np.sin(theta + np.pi / 2))),
             (int(x + r * np.cos(theta + np.pi / 2)), int(y - r * np.sin(theta + np.pi / 2))),
             color,
             THICKNESS)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'B',
                (int(x + r / 2 * np.cos(theta + np.pi / 4)), int(y - r / 2 * np.sin(theta + np.pi / 4))),
                font, 0.8, color, THICKNESS, cv2.LINE_AA)
    cv2.putText(img, 'W',
                (int(x + r / 2 * np.cos(theta + 3 * np.pi / 4)), int(y - r / 2 * np.sin(theta + 3 * np.pi / 4))),
                font, 0.8, color, 2, cv2.LINE_AA)


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


if __name__ == "__main__":
    main()
