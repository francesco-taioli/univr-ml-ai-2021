import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import cv2
import albumentations as A

def augment(image):
    """
    :param image: rgb image (dtype uint8)
    :return: augmented image (dtype uint8)
    """
    # shape = (image.shape[0], image.shape[1])
    probability = int(random.random() * 100)
    # only 50% of the images will be affected by data augmentation
    if probability<50:
        image = cv2noise(image)
        image = cv2brightness(image)

    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.2),
        A.RandomContrast(p=0.2),
        A.RandomGamma(p=0.2),
        A.HueSaturationValue(p=0.2),
    ])

    transformed = transform(image=image)
    image = transformed["image"]

    return image

def cv2brightness (input_image):
    """
    :param image: rgb image (dtype uint8)
    :return: rgb image with altered brightness (50% of cases)
    """
    image = np.copy(input_image)
    probability = int(random.random() * 100)
    if 0 <= probability < 25:
        brtadj = int(random.random() * 15)
        image[image < 255 - brtadj] += brtadj
    elif 75 < probability <= 100:
        brtadj = int(random.random() * 15)
        image[image > 0 + brtadj] -= brtadj
    return image


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def cv2noise (input_image):
    """
    :param image: rgb image (dtype uint8)
    :return: rgb image with added noise (50% of cases)
    """
    image = np.copy(input_image)
    probability = int(random.random() * 100)
    if 0 <= probability < 50:
        add = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8)
        cv2.randn(add, np.asarray([0, 0, 0]), np.asarray([5, 5, 5]))
        image = image + add
        image[image > 255] = 255
        image[image < 0] = 0

    return image