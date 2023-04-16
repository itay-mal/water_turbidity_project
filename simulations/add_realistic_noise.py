import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# image_path = './statistic_test/5.png'
images_root = './statistic_test'
images_target = './statistic_test_noised'
n_well = 32870  # TODO: replace with correct value
quantum_eff = .7  # TODO: replace with correct value
sigma_n = 6.2  # TODO: replace with correct value
show = False


def main():
    for fname in os.listdir(images_root):
        if not fname.endswith('.png'): continue
        image_path = os.path.join(images_root, fname)
        im = addd_noise(image_path)
        if show:
            print(np.max(im))
            plt.imshow(im)
            plt.show()
        else:
            if not os.path.isdir(images_target):
                os.mkdir(images_target)
            plt.imsave(os.path.join(images_target, fname), im)


def addd_noise(image_path):
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # [R,G,B]
    im = im.astype(np.float64)
    im *= n_well  # [electrons]
    im /= quantum_eff  # [photons]
    im += im + np.sqrt(im)*np.random.randn(im.shape[0], im.shape[1], im.shape[2])  # add photon noise
    im *= quantum_eff  # [electrons]
    im += np.sqrt(sigma_n)*np.random.randn(im.shape[0], im.shape[1], im.shape[2])  # add electronic noise
    im /= n_well  # [R,G,B]
    im = im.astype(np.uint8)
    return im


if __name__ == '__main__':
    main()
