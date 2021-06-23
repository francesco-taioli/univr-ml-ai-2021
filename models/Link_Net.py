import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from segmentation_models import Linknet

def Link_Net(input_shape=(256,256,3)):
    BACKBONE = 'vgg16'
    return Linknet(backbone_name=BACKBONE,
                   input_shape=input_shape,
                   classes=3,
                   activation='softmax',
                   encoder_weights=None
                   )