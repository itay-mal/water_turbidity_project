import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

image_path = './dataset_for_segmentation/73.png'
# images_root = './statistic_test'
images_target = './same_image_noised'
num_of_copies = 100
n_well = 32870  # TODO: replace with correct value
gamma = n_well/256
quantum_eff = .7  # TODO: replace with correct value
sigma_n = 6.2  # TODO: replace with correct value
show = False


def main():
    noise_same_render()

def noise_same_render():
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # [R,G,B]
    for i in tqdm(range(num_of_copies)):
        im_noised = add_noise(im)
        if show:
            print(np.max(im_noised))
            plt.imshow(im_noised)
            plt.show()
        else:
            if not os.path.isdir(images_target):
                os.mkdir(images_target)
            plt.imsave(os.path.join(images_target, "{:03}.png".format(i)), im_noised)

def noise_different_renderings():
    for fname in os.listdir(images_root):
        if not fname.endswith('.png'): continue
        image_path = os.path.join(images_root, fname)
        im = cv2.imread(image_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # [R,G,B]
        im = add_noise(im)
        if show:
            print(np.max(im))
            plt.imshow(im)
            plt.show()
        else:
            if not os.path.isdir(images_target):
                os.mkdir(images_target)
            plt.imsave(os.path.join(images_target, fname), im)


def add_noise(im):
    im = im.astype(np.float64)
    im *= gamma  # [electrons]
    im /= quantum_eff  # [photons]
    im += np.sqrt(im)*np.random.randn(im.shape[0], im.shape[1], im.shape[2])  # add photon noise
    # TODO: find picture statistics here, for sanity check
    # TODO: compare with torch poisson and modified normal noise
    im *= quantum_eff  # [electrons]
    im += sigma_n*np.random.randn(im.shape[0], im.shape[1], im.shape[2])  # add electronic noise
    im /= gamma  # [R,G,B]
    im = np.clip(im, 0, 255) # avoid overflow
    im = im.astype(np.uint8)
    return im


if __name__ == '__main__':
    main()
