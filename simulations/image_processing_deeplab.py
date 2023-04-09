import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

n_air = 1
n_water = 1.33
focal = 20e-3
target_r = 0.15
sensor_size = 24e-3
num_targets = 2
# img_path = '/home/itay.mal/Downloads/football1.jpeg'
# img_path = './variable_dists_and_sigma_s/5.png'
img_path = './texture_test.png'


def main():
    model = torch.hub.load('pytorch/vision:v0.10.0',
                           'deeplabv3_mobilenet_v3_large',
                           weights='DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT')
    model.eval()
    input_image = Image.open(img_path)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)
    plt.figure()
    plt.imshow(r)
    plt.figure()
    plt.imshow(input_image)
    plt.show()


if __name__ == "__main__":
    main()
