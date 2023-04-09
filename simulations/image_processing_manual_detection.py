import cv2
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

N_AIR = 1
N_WATER = 1.33
FOCAL = 20e-3
TARGET_R = 0.15
SENSOR_SIZE = 24e-3
num_targets = 2
img_path = "./dataset_for_segmentation/17.png"
THICKNESS = 2


def main():
    img = cv2.imread(img_path)
    img_clean = cv2.imread(img_path)
    cv2.namedWindow('BGR')

    def null(x):
        pass

    cv2.createTrackbar("T1_X", "BGR", 0, img.shape[1], null)
    cv2.createTrackbar("T1_Y", "BGR", 0, img.shape[0], null)
    cv2.createTrackbar("T1_R", "BGR", 0, int(min(img.shape[:1]) / 2), null)
    cv2.createTrackbar("T1_Theta", "BGR", 0, 180, null)
    cv2.createTrackbar("T2_X", "BGR", 0, img.shape[1], null)
    cv2.createTrackbar("T2_Y", "BGR", 0, img.shape[0], null)
    cv2.createTrackbar("T2_R", "BGR", 0, int(min(img.shape[:1]) / 2), null)
    cv2.createTrackbar("T2_Theta", "BGR", 0, 180, null)

    while True:
        # refresh image
        img = cv2.imread(img_path)

        # read trackbar values
        x1 = cv2.getTrackbarPos('T1_X', 'BGR')
        y1 = cv2.getTrackbarPos('T1_Y', 'BGR')
        r1 = cv2.getTrackbarPos('T1_R', 'BGR')
        theta1 = cv2.getTrackbarPos('T1_Theta', 'BGR') / 180 * np.pi
        x2 = cv2.getTrackbarPos('T2_X', 'BGR')
        y2 = cv2.getTrackbarPos('T2_Y', 'BGR')
        r2 = cv2.getTrackbarPos('T2_R', 'BGR')
        theta2 = cv2.getTrackbarPos('T2_Theta', 'BGR') / 180 * np.pi

        # draw target cursors
        draw_target_shape(img, x1, y1, r1, theta1, (255, 0, 0))
        draw_target_shape(img, x2, y2, r2, theta2, (0, 255, 0))

        # draw annotated image
        cv2.imshow('BGR', img)

        # exit point
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # calculate targets distance and print to console
        if key == ord('c'):
            # test_img = np.zeros()
            # t1_dist = calc_distance(img.shape[0], r1)
            # t2_dist = calc_distance(img.shape[0], r2)
            t1_w = []
            t1_b = []
            t2_w = []
            t2_b = []
            for y_hat, x_hat in product(range(img.shape[0]), range(img.shape[1])):
                if np.linalg.norm(np.array([x1, y1]) - np.array([x_hat, y_hat])) < r1:
                    angle = get_angle((x_hat - x1, y_hat - y1))
                    if (((theta1 <= angle) and (angle <= theta1 + np.pi / 2))
                            or ((theta1 + np.pi <= angle) and (angle <= theta1 + 3 * np.pi / 2))):
                        t1_b.append(img_clean[y_hat, x_hat])
                    else:
                        t1_w.append(img_clean[y_hat, x_hat])
            print(len(t1_w))
            # print(f'target1 distance: {t1_dist}[m] | target2 distance: {t2_dist}[m]')


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def get_angle(v1):
    """ Returns the angle in radians of vector 'v1':

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector((1, 0))
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if v1_u[1] < 0:
        angle += np.pi
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
