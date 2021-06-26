import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from segmentation_models import PSPNet, Unet

# HEIGTH and WIDTH should be divisible by 6 * downsample factor
def PSP_Net(input_shape=(256,256,3)):
    BACKBONE = 'resnet34'
    return PSPNet(BACKBONE,
                  input_shape=input_shape,
                  classes=3,
                  encoder_weights= None, #'imagenet',
                  downsample_factor=4,
                  psp_conv_filters=1024
                  )

# 40/40 [==============================] - 20s 508ms/step - loss: 0.5887 -
# tversky_loss: 2.1310 - mean_IoU: 0.3952 - pixel_accuracy: 0.8882 - lr: 1.0000e-03 -
# val_loss: 0.5581 - val_tversky_loss: 2.0523 - val_mean_IoU: 0.5580 - val_pixel_accuracy: 0.9515 - val_lr: 0.0010

