import cv2
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import os

N_AIR = 1
N_WATER = 1  # 1.33
FOCAL = 20e-3
TARGET_R = 0.15
SENSOR_SIZE = 24e-3
num_targets = 2
# img_path = "./same_att_ceoff.exr"
# img_path = "./attenuation_coeffs_sweep_no_ior/006.exr"
img_path = "D:/Desktop/001.exr"
# img_path = "./dataset_for_segmentation/87.png"
# img_path = "./white_light_test.png"
THICKNESS = 1


def main():
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = '1'
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    img_clean = cv2.imread(img_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    cv2.namedWindow('BGR')
    cv2.namedWindow('controls', cv2.WINDOW_NORMAL)

    def null(x):
        pass

    cv2.createTrackbar("T1_X", "controls", 271, img.shape[1], null)
    cv2.createTrackbar("T1_Y", "controls", 250, img.shape[0], null)
    cv2.createTrackbar("T1_R", "controls", 104, int(min(img.shape[:1]) / 2), null)
    cv2.createTrackbar("T1_Theta", "controls", 0, 180, null)
    cv2.createTrackbar("T2_X", "controls", 426, img.shape[1], null)
    cv2.createTrackbar("T2_Y", "controls", 250, img.shape[0], null)
    cv2.createTrackbar("T2_R", "controls", 52, int(min(img.shape[:1]) / 2), null)
    cv2.createTrackbar("T2_Theta", "controls", 0, 180, null)
    cv2.createTrackbar("T3_X", "controls", 375, img.shape[1], null)
    cv2.createTrackbar("T3_Y", "controls", 100, img.shape[0], null)
    cv2.createTrackbar("T3_R", "controls", 30, int(min(img.shape[:1]) / 2), null)

    while True:
        # refresh image
        img = (cv2.imread(img_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))**0.2

        # read trackbar values
        x1 = cv2.getTrackbarPos('T1_X', 'controls')
        y1 = cv2.getTrackbarPos('T1_Y', 'controls')
        r1 = cv2.getTrackbarPos('T1_R', 'controls')
        theta1 = cv2.getTrackbarPos('T1_Theta', 'controls') / 180 * np.pi
        x2 = cv2.getTrackbarPos('T2_X', 'controls')
        y2 = cv2.getTrackbarPos('T2_Y', 'controls')
        r2 = cv2.getTrackbarPos('T2_R', 'controls')
        theta2 = cv2.getTrackbarPos('T2_Theta', 'controls') / 180 * np.pi
        x3 = cv2.getTrackbarPos('T3_X', 'controls')
        y3 = cv2.getTrackbarPos('T3_Y', 'controls')
        r3 = cv2.getTrackbarPos('T3_R', 'controls')

        # draw target cursors
        draw_target_shape(img, x1, y1, r1, theta1, (255, 0, 0))
        draw_target_shape(img, x2, y2, r2, theta2, (0, 255, 0))
        draw_background_target_shape(img, x3, y3, r3, (0, 0, 255))

        # draw annotated image
        cv2.imshow('BGR', img)

        # exit point
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # calculate targets distance and print to console
        if key == ord('c'):
            test_img = np.zeros(img.shape)
            t1_dist = calc_distance(img.shape[0], r1)
            t2_dist = calc_distance(img.shape[0], r2)
            t1_w = []
            t1_b = []
            t2_w = []
            t2_b = []
            t3 = []

            def in_target(x, y, r, x_hat_, y_hat_):
                """check if given pixel (x_hat, y_hat) is inside a target with center coords (x,y) and radius r"""
                return np.linalg.norm(np.array([x, y]) - np.array([x_hat_, y_hat_])) < r

            def in_black_sections(theta, angle_):
                """check if the angle is in the correct ranges to be in the black sections of the target"""
                if 0 < theta <= np.pi/2:
                    return np.tan(theta) <= np.tan(angle_) or np.tan(angle_) <= np.tan(theta + np.pi / 2)
                return np.tan(theta) <= np.tan(angle_) <= np.tan(theta + np.pi / 2)

            # collect pixels for targets' sections
            for y_hat, x_hat in product(range(img.shape[0]), range(img.shape[1])):
                if in_target(x1, y1, r1, x_hat, y_hat): # in target 1
                    angle = get_angle((x_hat - x1, y1 - y_hat))  # flipped on Y axis bc img coord sys is upside down
                    if in_black_sections(theta1, angle):
                        t1_b.append(img_clean[y_hat, x_hat])
                        test_img[y_hat, x_hat] = (255, 0, 0)
                    else:
                        t1_w.append(img_clean[y_hat, x_hat])
                        test_img[y_hat, x_hat] = (0, 255, 0)

                elif in_target(x2, y2, r2, x_hat, y_hat):  # in target 2
                    angle = get_angle((x_hat - x2, y2 - y_hat))  # flipped on Y axis bc img coord sys is upside down
                    if in_black_sections(theta2, angle):
                        t2_b.append(img_clean[y_hat, x_hat])
                        test_img[y_hat, x_hat] = (255, 255, 0)
                    else:
                        t2_w.append(img_clean[y_hat, x_hat])
                        test_img[y_hat, x_hat] = (0, 255, 255)

                elif in_target(x3, y3, r3, x_hat, y_hat):  # in target 3
                    t3.append(img_clean[y_hat, x_hat])
                    test_img[y_hat, x_hat] = (0, 0, 255)

            cv2.namedWindow('TEST IMAGE')
            cv2.imshow('TEST IMAGE', test_img)
            print(f'target1 distance: {t1_dist}[m] | target2 distance: {t2_dist}[m]')
            t1_b = np.array(t1_b)
            t1_w = np.array(t1_w)
            t2_b = np.array(t2_b)
            t2_w = np.array(t2_w)
            clac_attenuation_coeffs(t1_dist, t1_w, t1_b, t2_dist, t2_w, t2_b, t3)


def clac_attenuation_coeffs(t1_dist, t1_w, t1_b, t2_dist, t2_w, t2_b, t3):
    t1_w_avg = np.average(t1_w, axis=0)
    t1_b_avg = np.average(t1_b, axis=0)
    t2_w_avg = np.average(t2_w, axis=0)
    t2_b_avg = np.average(t2_b, axis=0)
    t3_avg = np.average(t3, axis=0)

    print("t1[B,G,R]:", t1_w_avg, "t2[B,G,R]:", t2_w_avg)
    att_B_w = - np.log((t1_w_avg[0])/(t2_w_avg[0])) / (t1_dist - t2_dist)
    att_G_w = - np.log((t1_w_avg[1])/(t2_w_avg[1])) / (t1_dist - t2_dist)
    att_R_w = - np.log((t1_w_avg[2])/(t2_w_avg[2])) / (t1_dist - t2_dist)
    # att_B_w = - np.log((t1_w_avg[0] - t3_avg[0])/(t2_w_avg[0] - t3_avg[0])) / (t1_dist - t2_dist)
    # att_G_w = - np.log((t1_w_avg[1] - t3_avg[1])/(t2_w_avg[1] - t3_avg[1])) / (t1_dist - t2_dist)
    # att_R_w = - np.log((t1_w_avg[2] - t3_avg[2])/(t2_w_avg[2] - t3_avg[2])) / (t1_dist - t2_dist)
    print(f'attenuation blue on white: {att_B_w}')
    print(f'attenuation green on white: {att_G_w}')
    print(f'attenuation red on white: {att_R_w}')


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

def draw_background_target_shape(img, x, y, r, color):
    cv2.circle(img, (x, y), r, color, THICKNESS)




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
