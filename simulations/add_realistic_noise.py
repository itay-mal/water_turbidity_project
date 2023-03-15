import numpy as np
import cv2
import matplotlib.pyplot as plt

image_path = './statistic_test/5.png'
n_well = 10000.0  # TODO: replace with correct value
quantum_eff = .5  # TODO: replace with correct value
sigma_n = 30.0  # TODO: replace with correct value


def main():
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
    print(np.max(im))
    plt.imshow(im)
    plt.show()


if __name__ == '__main__':
    main()
