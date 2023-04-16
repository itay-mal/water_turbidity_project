import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


imgs_path = "./statistic_test_noised"


def calc_mean(imgs):
    """
    calculate mean per pixel across all images
    :param imgs: np.array (imH x imW x N) where N is the # of images
    :return: np.array (imH x im W) with mean per pixel
    """
    return np.mean(imgs, axis=2)


def cal_variance(imgs):
    """
    calculate variance per pixel across all images
    :param imgs: np.array (imH x imW x N) where N is the # of images
    :return: np.array (imH x im W) with variance per pixel
    """
    return np.var(imgs, axis=2)


def plot_var_vs_mean(mean, var):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x=mean, y=var, s=0.5)
    ax.set_title('Variance vs Mean')
    ax.set_xlabel('Mean')
    ax.set_ylabel('Variance')
    pass


def main():
    imgs = read_imgs_gray(imgs_path)
    mean_mat = calc_mean(imgs)
    display_image(mean_mat, 'Mean')
    var_mat  = cal_variance(imgs)
    display_image(var_mat, 'Variance')
    plot_var_vs_mean(mean=mean_mat.flatten(), var=var_mat.flatten())
    plt.show()


def read_imgs_gray(path):
    """
    read stacked matrix of the images
    :param path: directory for images to read
    :return: imgs: np.array (imH X imW X N) where N in the number of images, containing grayscale images
    """
    imgs_list = []
    for p in os.listdir(path):
        if not p.endswith(".png"): continue
        imgs_list.append(cv2.cvtColor(cv2.imread(os.path.join(path, p)), cv2.COLOR_BGR2GRAY))
    imgs = np.stack(imgs_list, axis=2)
    return imgs


def display_image(img, title: str):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img, cmap="gray")
    ax.set_title(title)


if __name__ == "__main__":
    main()
