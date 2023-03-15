import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

img = cv2.imread("tmp5.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# cv2.imwrite("pic_test.png", img)

hist_r = plt.hist(img_rgb[:, :, 0], 256)
plt.show()
