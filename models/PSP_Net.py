import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from segmentation_models import PSPNet, Unet

# HEIGTH and WIDTH should be divisible by 6 * downsample factor
def PSP_Net(input_shape=(256,256,3)):
    BACKBONE = 'resnet34'
    return Unet(BACKBONE,
                  input_shape=input_shape,
                  classes=3,
                  encoder_weights= None, #'imagenet',
                  # downsample_factor=4
                  )

