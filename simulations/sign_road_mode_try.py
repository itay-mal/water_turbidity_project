import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import matplotlib.pyplot as plt

def normalize(im):
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im - imagenet_stats[0]) / imagenet_stats[1]


class BB_model(nn.Module):
    def __init__(self):
        super(BB_model, self).__init__()
        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())[:8]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x), self.bb(x)


def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[1], bb[0]), bb[3] - bb[1], bb[2] - bb[0], color=color,
                         fill=False, lw=3)


def show_corner_bb(im, bb):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))


# path = 'road_sign_detection_input/images/road700.png'
path = './dataset_for_segmentation/4031.png'
model = torch.load('sign_detection_weights.pt')

img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32)
img /= 255
resized_img = cv2.resize(img, (284, 284))
normalized_img = normalize(resized_img).astype(np.float32)
xx = ((torch.tensor(normalized_img)).permute((2, 0, 1))).unsqueeze(0)

print(xx.shape)

out_class, out_bb = model(xx)

bb_hat = out_bb.detach().cpu().numpy()
bb_hat = bb_hat.astype(int)
show_corner_bb(resized_img, bb_hat[0])
plt.show()
