from segmentation_models import PSPNet, Unet
from utils import get_env_variable

# HEIGTH and WIDTH should be divisible by 6 * downsample factor
class PSP_Net():
    def __init__(self):
        self.WIDTH = int(get_env_variable('WIDTH'))
        self.HEIGHT = int(get_env_variable('HEIGHT'))
        self.BACKBONE = 'resnet34'

    def get_model(self):
        return PSPNet(self.BACKBONE,
                      input_shape=(self.HEIGHT, self.WIDTH, 3),
                      classes=3,
                      encoder_weights='imagenet',
                      downsample_factor=8
                      )
