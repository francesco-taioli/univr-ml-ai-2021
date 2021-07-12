import random
import numpy as np
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
    transform = A.Compose([
        A.RandomBrightnessContrast(p=prob),
        A.RandomContrast(p=prob),
        A.RandomGamma(p=prob),
        A.HueSaturationValue(p=prob),
        A.GaussNoise(p=prob)
        # A.CLAHE(p=prob)
    ])

    transformed = transform(image=image)
    return transformed["image"]


# Showing how the augmentation is done (5 images augmented with their original counterparts
# if you want to see an example, run python augmentation.py
# you have to first run the main.py in order to generate the set_imgs file
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

        ax.append(fig.add_subplot(rows, columns, i + 1))
        image = dataset[i]
        plt.imshow(image.astype(np.uint8))
        plt.axis('off')

        if p < 50:
            dataset[i] = cv2.flip(dataset[i], -1)

        ax.append(fig.add_subplot(rows, columns, columns + i + 1))
        image = augment(dataset[i], prob=0.6)
        plt.imshow(image.astype(np.uint8))
        plt.axis('off')

    fig.suptitle('Augmentation examples', fontsize=16)
    ax[4].set_title('Original images')
    ax[5].set_title('Augmented images')
    plt.show()
