from segmentation_models import PSPNet

# define model
# todo change input size
class PSP_Net():
    def __init__(self):
        self.BACKBONE = 'resnet34'

    def get_model(self):
        return  PSPNet(self.BACKBONE,
                          input_shape=(256, 256, 3),
                          classes=3,
                          encoder_weights='imagenet'
                          )