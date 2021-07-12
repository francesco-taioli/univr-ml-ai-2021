import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from skimage import color
from utils import get_env_variable


def pixel_wise_loss():
    pos_weight = tf.constant([[0.1, 2.0, 2.0]])

    def pwl(y_true, y_pred):
        loss = tf.nn.weighted_cross_entropy_with_logits(
            y_true,
            y_pred,
            pos_weight,
            name=None
        )
        return K.mean(loss, axis=-1)

    return pwl


def weighted_categorical_crossentropy():
    weights = [1.0, 2.0, 2.0]

    def wcce(y_true, y_pred):
        Kweights = K.constant(weights)
        # if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)

    return wcce


def mean_IoU(y_true, y_pred):
    '''
    https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanIoU
    Implementation of the MeanIou
    '''
    number_classes = 3
    # print(y_true.shape) #(8, 256, 256, 3)
    # print(y_pred.shape) #(8, 256, 256, 3)

    eps = 1e-6
    number_items_in_batches = y_true.shape[0]
    IoU_mean = 0
    for b in range(number_items_in_batches):
        IoU_channel = 0
        pred = np.argmax(y_pred[b, :, :], axis=-1)
        mask_pred = np.zeros((int(get_env_variable('HEIGHT')), int(get_env_variable('WIDTH')), 3), dtype=int)
        for c in range(number_classes):
            # create mask
            mask_pred[:, :, c] = pred == c
            mask_pred_one_channel = mask_pred[:, :, c]
            mask = y_true[b, :, :, c].numpy().astype(int)  # 256 x 256

            # IOU = true_positive / (true_positive + false_positive + false_negative).

            true_positive = (mask & mask_pred_one_channel).sum()

            false_negative = (mask - mask_pred_one_channel)
            false_negative[false_negative == -1] = 0
            false_negative = false_negative.sum()

            false_positive = (mask_pred_one_channel - mask)
            false_positive[false_positive == -1] = 0
            false_positive = false_positive.sum()

            IoU_channel += true_positive / (true_positive + false_positive + false_negative + eps)
        IoU_mean += IoU_channel / number_classes
    return IoU_mean / number_items_in_batches


def pixel_accuracy(y_true, y_pred):
    '''
    Implementation of the pixel accuracy metric
    '''
    number_items_in_batches = y_true.shape[0]
    sum_true = 0
    sum_total = 0

    for b in range(number_items_in_batches):
        pred_mask = np.argmax(y_pred[b], axis=-1)
        real_mask = np.argmax(y_true[b], axis=-1)

        sum_true += (pred_mask == real_mask).sum()
        sum_total += int(get_env_variable('HEIGHT')) * int(get_env_variable('WIDTH'))

    return sum_true / sum_total


def overlay_prediction(img, prediction):
    """
    return the image with the predicted segmentation parts overlayed on it
    """
    alpha = 0.6
    color_mask = prediction
    color_mask[:, :, 0] = 0  # delete the background

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked


def predict_mask_and_plot(img, mask, model, epoch=0, save=False, index=-1):
    image = np.reshape(img, newshape=(1, img.shape[0], img.shape[1], img.shape[2]))  # / 255.
    WIDTH = int(get_env_variable('WIDTH'))
    HEIGHT = int(get_env_variable('HEIGHT'))
    pred = model.predict(image)

    res = np.argmax(pred[0], axis=-1)
    final = np.zeros((HEIGHT, WIDTH, 3))
    final[:, :, 0] = res == 0
    final[:, :, 1] = res == 1
    final[:, :, 2] = res == 2

    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(20, 6)
    axs[0].imshow(img), axs[0].set_title('Original Image')
    axs[1].imshow(overlay_prediction(img, mask * 255)), axs[1].set_title('True mask overlay')
    axs[2].imshow(overlay_prediction(img, final)), axs[2].set_title('Final result overlay')

    if save and index == -1:
        plt.savefig(os.path.join(get_env_variable('TRAIN_DATA'), 'images', 'epoch{}.png'.format(epoch)))
    elif save and index != -1:
        plt.savefig(os.path.join(get_env_variable('TRAIN_DATA'), 'images', '{}.png'.format(index)))
    else:
        plt.show()


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


class Show_Intermediate_Pred(tf.keras.callbacks.Callback):

    def __init__(self, image, mask):
        self.image = image
        self.mask = mask

    def on_epoch_end(self, epoch, logs=None):
        predict_mask_and_plot(self.image, self.mask, self.model, epoch, save=True)
