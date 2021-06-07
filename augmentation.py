import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import cv2
import albumentations as A
import pickle
import os
import matplotlib.pyplot as plt

def augment(image, prob=0.2):
    """
    :param image: rgb image (dtype uint8)
    :return: augmented image (dtype uint8)
    """
    # shape = (image.shape[0], image.shape[1])
    probability = int(random.random() * 100)
    # only 50% of the images will be affected by data augmentation

    transform = A.Compose([
        A.RandomBrightnessContrast(p=prob),
        A.RandomContrast(p=prob),
        A.RandomGamma(p=prob),
        A.HueSaturationValue(p=prob),
        A.GaussNoise(p=prob)
    ])

    transformed = transform(image=image)
    image = transformed["image"]

    return image

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

if __name__ == "__main__":

    with open(os.path.join('dataset', 'set_imgs'), 'rb') as imgs:
            dataset = pickle.load(imgs)

    rows = 2
    columns = 5
    fig = plt.figure()
    ax = []
    # 5 augmented images to show how the augmentation is performed (original and augmented below)

    for i in range(columns):

        p = random.random() * 100 // 1

        if p < 50:
            dataset[i] = cv2.flip(dataset[i], -1)

        ax.append(fig.add_subplot(rows, columns, i + 1))
        image = dataset[i]
        plt.imshow(image.astype(np.uint8))
        plt.axis('off')

        ax.append(fig.add_subplot(rows, columns, columns + i + 1))
        image = augment(dataset[i], prob=0.4)
        plt.imshow(image.astype(np.uint8))
        plt.axis('off')

    plt.show()