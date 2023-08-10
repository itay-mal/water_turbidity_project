import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from itertools import product
from sklearn.metrics import r2_score
import os
from tqdm import tqdm

N_AIR = 1
N_WATER = 1.33
FOCAL = 20e-3
TARGET_R = 0.15
SENSOR_SIZE = 24e-3 # for 35mm sensor: 24X36mm
num_targets = 2


#### consts from manual GUI ####
T1_X = 271 # 271
T1_Y = 250
T1_R = 104 #104
T1_DIST = 0.6 #0.6
T1_THETA = 0
T2_X = 426 #405 #426
T2_Y = 250
T2_R = 52 #31 #52
T2_DIST = 1.2# 2 #1.2
T2_THETA = 0
#############################



images_root = "./absorption_emitter_in_water"
img_path = "C:/Users/nitay/Desktop/000.png"
# img_path = "./white_light_test.png"
THICKNESS = 1


def main():
    coeffs = []
    light_arr = []
    floor_ref = []

    with open(os.path.join(images_root, "log.txt"), 'rt') as f:
        for line in tqdm(f.readlines()):
            idx, d1, d2, sigma_s, sigma_a, _, light, floor = line.split(':')
            idx = int(idx)
            d1 = eval(d1.split('=')[-1])
            d2 = eval(d2.split('=')[-1])
            sigma_a = eval(sigma_a.split('=')[-1])
            sigma_s = eval(sigma_s.split('=')[-1])
            light = eval(light.split('=')[-1])
            floor = eval(floor.split('=')[-1])
            img = cv2.imread(os.path.join(images_root, '{:03d}.exr'.format(idx)), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_UNCHANGED)
            coeffs.append([get_att_coeffs(img, d1, d2)])
            coeffs[idx].append((sigma_a[0] + sigma_s[0], sigma_a[1] + sigma_s[1], sigma_a[2] + sigma_s[2]))
            light_arr.append(light)
            floor_ref.append(floor)

    plot_calculated_vs_expected_with_LR(coeffs)
    # plot_calculated_vs_expected_normalized(coeffs)
    # show_img_calculated_vs_expected(cv2.imread(os.path.join(images_root, '000.exr'), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_UNCHANGED),coeffs[0][1], coeffs[0][0])


def show_img_calculated_vs_expected(img, expected, calc):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('no floor')
    plt.axis('off')
    plt.figtext(0.5, 0.01, 'expected = {}\ncalculated = {}'.format(expected, tuple((round(i, 4) for i in calc))), ha='center', fontsize=14, bbox={'facecolor':'orange','alpha':0.5,'pad':5})
    ax.imshow(img)
    plt.savefig('no_floor_img_coeffs.png')

def plot_calculated_vs_expected_normalized(coeffs):
    coeffs = np.array(coeffs)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x=coeffs[:,1,0] - coeffs[0,1,0], y=coeffs[:,0,0] - coeffs[0,0,0], s=0.5, c='tab:red', label='red channel')
    ax.scatter(x=coeffs[:,1,1] - coeffs[0,1,1], y=coeffs[:,0,1] - coeffs[0,0,1], s=0.5, c='tab:green', label='green channel')
    ax.scatter(x=coeffs[:,1,2] - coeffs[0,1,2], y=coeffs[:,0,2] - coeffs[0,0,2], s=0.5, c='tab:blue', label='blue channel')
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(-0.01, 1)
    ax.set_aspect('equal', 'box')
    ax.legend()
    ax.set_title('centered calculated VS expected attenuation coefficients')
    ax.set_ylabel('calculated attenuation [1/m]')
    ax.set_xlabel('expected attenuation [1/m]')
    plt.savefig('calculated_vs_expected_scatter_with_absorption_normalized.png')

def plot_calculated_vs_expected_with_LR(coeffs):
    coeffs = np.array(coeffs)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x=coeffs[:,1,0], y=coeffs[:, 0, 0], s=0.5, c='tab:red', label='red channel')
    linear_regressor_red = LinearRegression(fit_intercept=True)
    linear_regressor_red.fit((coeffs[:, 1, 0]).reshape(-1, 1),
                             (coeffs[:, 0, 0]).reshape(-1, 1))
    red_pred = linear_regressor_red.predict((coeffs[:, 1, 0]).reshape(-1, 1))
    ax.plot(coeffs[:, 1, 0],
            red_pred,
            c='tab:red',
            alpha=0.5,
            label=r"$R^2=$" + "{:.03f}\nslope={:.03f}".format(r2_score(coeffs[:, 0, 0], red_pred), linear_regressor_red.coef_[0, 0]))
    ax.scatter(x=coeffs[:, 1, 1], y=coeffs[:, 0, 1], s=0.5, c='tab:green', label='green channel')
    linear_regressor_green = LinearRegression(fit_intercept=True)
    linear_regressor_green.fit((coeffs[:, 1, 1]).reshape(-1, 1),
                               (coeffs[:, 0, 1]).reshape(-1, 1))
    green_pred = linear_regressor_green.predict((coeffs[:, 1, 1]).reshape(-1, 1))
    ax.plot(coeffs[:, 1, 1],
            green_pred,
            c='tab:green',
            alpha=0.5,
            label=r"$R^2=$" + "{:.03f}\nslope={:.03f}".format(r2_score(coeffs[:, 0, 1], green_pred), linear_regressor_green.coef_[0, 0]))
    ax.scatter(x=coeffs[:, 1, 2], y=coeffs[:, 0, 2], s=0.5, c='tab:blue', label='blue channel')
    linear_regressor_blue = LinearRegression(fit_intercept=True)
    linear_regressor_blue.fit((coeffs[:, 1, 2]).reshape(-1, 1),
                              (coeffs[:, 0, 2]).reshape(-1, 1))
    blue_pred = linear_regressor_blue.predict((coeffs[:, 1, 2]).reshape(-1, 1))
    ax.plot(coeffs[:, 1, 2],
            blue_pred,
            c='tab:blue',
            alpha=0.5,
            label=r"$R^2=$" + " {:.03f}\nslope={:.03f}".format(r2_score(coeffs[:, 0, 2], blue_pred), linear_regressor_blue.coef_[0, 0]))
    ax.set_xlim(-0.01, 5)
    ax.set_ylim(-0.01, 5)
    ax.set_aspect('equal', 'box')
    ax.legend()
    ax.set_title('calculated VS expected attenuation coefficients')
    ax.set_ylabel('calculated attenuation [1/m]')
    ax.set_xlabel('expected attenuation [1/m]')
    plt.savefig('calculated_vs_expected_absorption_emitter_in_water.png')

def plot_calculated_vs_expected_with_LR_for_3(coeffs):
    coeffs = np.array(coeffs)
    fig = plt.figure(figsize=(8, 5))
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        ax.scatter(x=coeffs[i::3,1,0], y=coeffs[i::3, 0, 0], s=0.5, c='tab:red', label='red channel')
        linear_regressor_red = LinearRegression()
        linear_regressor_red.fit((coeffs[i::3, 1, 0]).reshape(-1, 1),
                                (coeffs[i::3, 0, 0]).reshape(-1, 1))
        red_pred = linear_regressor_red.predict((coeffs[i::3, 1, 0]).reshape(-1, 1))
        ax.plot(coeffs[i::3, 1, 0],
                red_pred,
                c='tab:red',
                alpha=0.5,
                label=r"$R^2=$" + "{:.03f}\nslpoe={:.03f}".format(r2_score(coeffs[i::3, 0, 0], red_pred), linear_regressor_red.coef_[0, 0]))
        ax.scatter(x=coeffs[i::3, 1, 1], y=coeffs[i::3, 0, 1], s=0.5, c='tab:green', label='green channel')
        linear_regressor_green = LinearRegression()
        linear_regressor_green.fit((coeffs[i::3, 1, 1]).reshape(-1, 1),
                                   (coeffs[i::3, 0, 1]).reshape(-1, 1))
        green_pred = linear_regressor_green.predict((coeffs[i::3, 1, 1]).reshape(-1, 1))
        ax.plot(coeffs[i::3, 1, 1],
                green_pred,
                c='tab:green',
                alpha=0.5,
                label=r"$R^2=$" + "{:.03f}\nslpoe={:.03f}".format(r2_score(coeffs[i::3, 0, 1], green_pred), linear_regressor_green.coef_[0, 0]))
        ax.scatter(x=coeffs[i::3, 1, 2], y=coeffs[i::3, 0, 2], s=0.5, c='tab:blue', label='blue channel')
        linear_regressor_blue = LinearRegression()
        linear_regressor_blue.fit((coeffs[i::3, 1, 2]).reshape(-1, 1),
                                  (coeffs[i::3, 0, 2]).reshape(-1, 1))
        blue_pred = linear_regressor_blue.predict((coeffs[i::3, 1, 2]).reshape(-1, 1))
        ax.plot(coeffs[i::3, 1, 2],
                blue_pred,
                c='tab:blue',
                alpha=0.5,
                label=r"$R^2=$" + " {:.03f}\nslpoe={:.03f}".format(r2_score(coeffs[i::3, 0, 2], blue_pred), linear_regressor_blue.coef_[0, 0]))
        ax.set_xlim(-0.01, 1.2)
        ax.set_ylim(-0.01, 1.2)
        ax.set_aspect('equal', 'box')
        ax.legend()
        ax.set_title(f'absorption = {coeffs[i, 1, :]}')
        ax.set_ylabel('calculated attenuation [1/m]')
        ax.set_xlabel('expected attenuation [1/m]')
    fig.set_title(f'calculated VS expected attenuation coefficients ')
    plt.savefig('calculated_vs_expected_3_absorptions_LR.png')


def plot_calculated_vs_expected(coeffs):
    coeffs = np.array(coeffs)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x=coeffs[:,1,0], y=coeffs[:,0,0], s=0.5, c='tab:red', label='red channel')
    ax.scatter(x=coeffs[:,1,1], y=coeffs[:,0,1], s=0.5, c='tab:green', label='green channel')
    ax.scatter(x=coeffs[:,1,2], y=coeffs[:,0,2], s=0.5, c='tab:blue', label='blue channel')
    ax.set_xlim(-0.01, 2.5)
    ax.set_ylim(-0.01, 2.5)
    ax.set_aspect('equal', 'box')
    ax.legend()
    ax.set_title('calculated VS expected attenuation coefficients')
    ax.set_ylabel('calculated attenuation [1/m]')
    ax.set_xlabel('expected attenuation [1/m]')
    plt.savefig('calculated_vs_expected_scatter_with_absorption.png')

def plot_calculated_vs_light(coeffs, light):
    coeffs = np.array(coeffs)
    light = np.float32(light)
    light = np.expand_dims(light, axis=1)
    light = cv2.cvtColor(light, cv2.COLOR_RGB2HSV)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x=coeffs[:,0,0], y=light[:, :, 0], s=50, marker='^', c='tab:red', label='red channel')
    ax.scatter(x=coeffs[:,0,1], y=light[:, :, 0], s=60, marker='+', c='tab:green', label='green channel')
    ax.scatter(x=coeffs[:,0,2], y=light[:, :, 0], s=20, c='tab:blue', label='blue channel')
    ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_aspect('equal', 'box')
    ax.legend()
    ax.set_title('attenuation VS sun hue')
    ax.set_xlabel('attenuation coeffs')
    ax.set_ylabel('sun hue')
    plt.savefig('calculated_vs_sun_hue.png')

def plot_calculated_vs_light_amplitude(coeffs, light):
    coeffs = np.array(coeffs)
    light = np.sum(light, axis=1)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x=coeffs[:,0,0], y=light, s=50, marker='^', c='tab:red', label='red channel')
    ax.scatter(x=coeffs[:,0,1], y=light, s=60, marker='+', c='tab:green', label='green channel')
    ax.scatter(x=coeffs[:,0,2], y=light, s=20, c='tab:blue', label='blue channel')
    ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_aspect('equal', 'box')
    ax.legend()
    ax.set_title('attenuation VS sun radiance sum')
    ax.set_xlabel('attenuation coeffs')
    ax.set_ylabel('radiance sum')
    plt.savefig('calculated_vs_sun_amp.png')

def plot_calculated_vs_floor_ref(coeffs, floor_ref):
    coeffs = np.array(coeffs)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(y=coeffs[:,0,0], x=floor_ref, s=50, marker='^', c='tab:red', label='red channel')
    ax.scatter(y=coeffs[:,0,1], x=floor_ref, s=60, marker='+', c='tab:green', label='green channel')
    ax.scatter(y=coeffs[:,0,2], x=floor_ref, s=20, c='tab:blue', label='blue channel')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # ax.set_aspect('equal', 'box')
    ax.legend()
    ax.set_title('attenuation VS floor reflectance')
    ax.set_xlabel('floor reflectance')
    ax.set_ylabel('attenuation coeffs')
    plt.savefig('calculated_vs_floor_ref.png')


def get_att_coeffs(img, d1, d2):
    t1_dist = d1
    t2_dist = d2
    r1 = calc_radius_by_distance(img.shape[0], d1)
    x1 = round((img.shape[1]/2) - r1)
    y1 = T1_Y
    y2 = T2_Y
    r2 = calc_radius_by_distance(img.shape[0], d2)
    x2 = round((img.shape[1]/2) + r2)
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
    return np.linalg.norm(np.array([x, y]) - np.array([x_hat_, y_hat_])) < 0.95*r

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
    :return: attenuation coeffs in RGB format
    """
    t1_w_avg = np.average(t1_w, axis=0)
    t1_b_avg = np.average(t1_b, axis=0)
    t2_w_avg = np.average(t2_w, axis=0)
    t2_b_avg = np.average(t2_b, axis=0)
    att_B_w = - np.log((t1_w_avg[0]-t1_b_avg[0])/(t2_w_avg[0]-t2_b_avg[0])) / (t1_dist - t2_dist)
    att_G_w = - np.log((t1_w_avg[1]-t1_b_avg[1])/(t2_w_avg[1]-t2_b_avg[1])) / (t1_dist - t2_dist)
    att_R_w = - np.log((t1_w_avg[2]-t1_b_avg[2])/(t2_w_avg[2]-t2_b_avg[2])) / (t1_dist - t2_dist)
    # att_B_w = - np.log((t1_w_avg[0])/(t2_w_avg[0])) / (t1_dist - t2_dist)
    # att_G_w = - np.log((t1_w_avg[1])/(t2_w_avg[1])) / (t1_dist - t2_dist)
    # att_R_w = - np.log((t1_w_avg[2])/(t2_w_avg[2])) / (t1_dist - t2_dist)
    # att_B_b = - np.log(t1_b_avg[0]/t2_b_avg[0]) / (t1_dist - t2_dist)
    # att_G_b = - np.log(t1_b_avg[1]/t2_b_avg[1]) / (t1_dist - t2_dist)
    # att_R_b = - np.log(t1_b_avg[2]/t2_b_avg[2]) / (t1_dist - t2_dist)
    # print(f't1_b_avg= {t1_b_avg} t2_b_avg= {t2_b_avg} t1_w_avg=: {t1_w_avg} t2_w_avg=: {t2_w_avg}')
    # print(f'attenuation blue on black: {att_B_b}')
    # print(f'attenuation blue average: {np.average([att_B_w,att_B_b])}')
    # print(f'attenuation green on white: {att_G_w}')
    # print(f'attenuation green on black: {att_G_b}')
    # print(f'attenuation green average: {np.average([att_G_w,att_G_b])}')
    # print(f'attenuation red on white: {att_R_w}')
    # print(f'attenuation red on black: {att_R_b}')
    # print(f'attenuation red average: {np.average([att_R_w,att_R_b])}')
    return att_R_w, att_G_w, att_B_w


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

def calc_radius_by_distance(img_height, dist):
    """
    calculate the distance from the camera to the targets in meters
    :param img_height: in pixels
    :param dist: dist to the target in meters
    :return: for each target return radius in pixels
    """
    focal_eff = FOCAL * (N_WATER / N_AIR)
    pix_d = SENSOR_SIZE / img_height
    return (focal_eff * TARGET_R) / (pix_d * dist)


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
