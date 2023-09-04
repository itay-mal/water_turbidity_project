import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data, io
from skimage.filters import gaussian, median
from skimage import exposure
from skimage.segmentation import active_contour
import cv2
import os
from numpy.linalg import inv

import concurrent.futures
import time

class RANSAC:
    def __init__(self, x_data, y_data, n):
        self.x_data = x_data
        self.y_data = y_data
        self.n = n
        self.d_min = 99999
        self.best_model = None

    def random_sampling(self):
        sample = []
        save_ran = []
        count = 0

        # get three points from data
        while True:
            ran = np.random.randint(len(self.x_data))

            if ran not in save_ran:
                sample.append((self.x_data[ran], self.y_data[ran]))
                save_ran.append(ran)
                count += 1

                if count == 3:
                    break

        return sample

    def make_model(self, sample):
        # calculate A, B, C value from three points by using matrix

        pt1 = sample[0]
        pt2 = sample[1]
        pt3 = sample[2]

        A = np.array([[pt2[0] - pt1[0], pt2[1] - pt1[1]], [pt3[0] - pt2[0], pt3[1] - pt2[1]]])
        B = np.array([[pt2[0] ** 2 - pt1[0] ** 2 + pt2[1] ** 2 - pt1[1] ** 2],
                      [pt3[0] ** 2 - pt2[0] ** 2 + pt3[1] ** 2 - pt2[1] ** 2]])
        inv_A = inv(A)

        c_x, c_y = np.dot(inv_A, B) / 2
        c_x, c_y = c_x[0], c_y[0]
        r = np.sqrt((c_x - pt1[0]) ** 2 + (c_y - pt1[1]) ** 2)

        return c_x, c_y, r

    def eval_model(self, model):
        d = 0
        c_x, c_y, r = model

        for i in range(len(self.x_data)):
            dis = np.sqrt((self.x_data[i] - c_x) ** 2 + (self.y_data[i] - c_y) ** 2)

            if dis >= r:
                d += dis - r
            else:
                d += r - dis

        return d

    def execute_ransac(self):
        # find best model
        for i in range(self.n):
            model = self.make_model(self.random_sampling())
            d_temp = self.eval_model(model)

            if self.d_min > d_temp:
                self.best_model = model
                self.d_min = d_temp

path = "C:/Users/nitay/Desktop/051.png"

def main():
    img = io.imread(path)
    img = rgb2gray(img)
    img = adjust_gamma(img, 0.3)

    im_h, im_w = img.shape
    rad = 0.25 * im_w

    s1 = np.linspace(0, 2 * np.pi, 400)
    r1 = 0.5 * im_h + rad * np.sin(s1)
    c1 = 0.25 * im_w + rad * np.cos(s1)
    init_1 = np.array([r1, c1]).T

    s2 = np.linspace(0, 2 * np.pi, 400)
    r2 = 0.5 * im_h + rad * np.sin(s2)
    c2 = 0.75 * im_w + rad * np.cos(s2)
    init_2 = np.array([r2, c2]).T

    snake1 = active_contour(gaussian(img, 3, preserve_range=False),
                           init_1, alpha=0.001, beta=15, gamma=0.001)
    snake2 = active_contour(gaussian(img, 3, preserve_range=False),
                           init_2, alpha=0.001, beta=15, gamma=0.001)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init_1[:, 1], init_1[:, 0], '--r', lw=3)
    ax.plot(init_2[:, 1], init_2[:, 0], '--r', lw=3)
    ax.plot(snake1[:, 1], snake1[:, 0], '-b', lw=3)
    ax.plot(snake2[:, 1], snake2[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])


    ransac1 = RANSAC(snake1[:, 1], snake1[:, 0], 50)
    ransac1.execute_ransac()
    a1, b1, r1 = ransac1.best_model[0], ransac1.best_model[1], ransac1.best_model[2]

    ransac2 = RANSAC(snake2[:, 1], snake2[:, 0], 50)
    ransac2.execute_ransac()
    a2, b2, r2 = ransac2.best_model[0], ransac2.best_model[1], ransac2.best_model[2]

    # show result
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    circle1 = plt.Circle((a1, b1), radius=r1, color='r', fc='y', fill=False)
    plt.gca().add_patch(circle1)
    circle2 = plt.Circle((a2, b2), radius=r2, color='g', fc='y', fill=False)
    plt.gca().add_patch(circle2)
    ax.set_aspect('equal', 'box')
    # ax.set_xlim([ 0, 749])
    # ax.set_ylim([ 499, 0]) #inverse y axis between graph and image
    plt.show()

def AC_detction(img, show_intermediate_results = False):
    img = rgb2gray(img)
    
    if show_intermediate_results:
        # show input image in grayscale
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.set_title("Original Image in Grayscale")

    # TODO: keep this thresholding?
    # img[img < 0.15] = 0
    # img[img > 0.8] = 1
    # img[((0.05 <= img)*(img <= 0.8))] = 0.5

    # if show_intermediate_results:
    #     fig, ax = plt.subplots(figsize=(7, 7))
    #     ax.imshow(img, cmap=plt.cm.gray)

    # initilazing snakes for active contours
    im_h, im_w = img.shape
    rad = 0.25*im_w

    s1 = np.linspace(0, 2 * np.pi, 400)
    r1 = 0.5  * im_h + rad * np.sin(s1)
    c1 = 0.25 * im_w + rad * np.cos(s1)
    init_1 = np.array([r1, c1]).T

    s2 = np.linspace(0, 2 * np.pi, 400)
    r2 = 0.5  * im_h + rad  * np.sin(s2)
    c2 = 0.75 * im_w + rad * np.cos(s2)
    init_2 = np.array([r2, c2]).T

    # shrink snakes with active contours algorithm
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        img_gaus = gaussian(img, 3, preserve_range=False)
        f1 = pool.submit(active_contour, img_gaus, init_1, alpha=0.0001, beta=20, gamma=0.0001)
        f2 = pool.submit(active_contour, img_gaus, init_2, alpha=0.0001, beta=20, gamma=0.0001)
        snake2 = f2.result()
        snake1 = f1.result()

    # snakes are returned as list in format (y,x), change to np.array in (x,y)
    snake1 = np.flip(np.array(snake1), axis=1)
    snake2 = np.flip(np.array(snake2), axis=1)

    if show_intermediate_results:
        # show active contours results
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(init_1[:, 1], init_1[:, 0], '--r', lw=3)
        ax.plot(init_2[:, 1], init_2[:, 0], '--r', lw=3)
        ax.plot(snake1[:, 0], snake1[:, 1], '-b', lw=3)
        ax.plot(snake2[:, 0], snake2[:, 1], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])
        ax.set_title("Active-Contours results")

    return snake1, snake2

if __name__ == "__main__":
    main()


