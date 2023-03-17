import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import yolov5
import tensorflow_hub as hub
import tensorflow as tf
n_air = 1
n_water = 1.33
focal = 20e-3
target_r = 0.15
sensor_size = 24e-3
num_targets = 2
# img_path = '/home/itay.mal/Downloads/football1.jpeg'
img_path = './other_images/Bone-marrow-smear-cells-myelocytes-cluster-metamyelocyte.webp'



def main():
    tf.compat.v1.disable_eager_execution()
    images = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # fy,fx = (192/images.shape[0], 192/images.shape[1])
    images = cv2.resize(images, (192, 192))
    images = np.expand_dims(images, axis=0)  # A batch of images with shape [batch_size, height, width, 3].
    module = hub.Module("https://kaggle.com/models/google/mobile-object-localizer-v1/frameworks/TensorFlow1/variations/object-detection-mobile-object-localizer-v1/versions/1")
    features = module(images, as_dict=True)  # Features with shape [batch_size, num_outputs].
    a = 1

if __name__ == "__main__":
    main()
