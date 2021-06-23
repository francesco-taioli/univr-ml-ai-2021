from tensorflow.keras.layers import Conv2D, MaxPooling2D, LayerNormalization, Softmax, \
    Conv2DTranspose, add, Activation, Reshape, BatchNormalization, Input
import  tensorflow as tf

def Seg_Net(shape, classes):
    inputs = Input(shape=shape)
    outputs = SegNetArchitecture(inputs, num_classes=classes)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def SegNetArchitecture(inputs, num_classes):

    conv1_1 = Conv2D(32, [3, 3], padding="same", activation="relu")(inputs)
    conv1_1 = LayerNormalization()(conv1_1)
    conv1_2 = Conv2D(32, [3, 3], padding="same", activation="relu")(conv1_1)
    conv1_2 = LayerNormalization()(conv1_2)
    pool1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="same")(conv1_2)

    conv2_1 = Conv2D(64, [3, 3], padding="same", activation="relu")(pool1)
    conv2_1 = LayerNormalization()(conv2_1)
    conv2_2 = Conv2D(64, [3, 3], padding="same", activation="relu")(conv2_1)
    conv2_2 = LayerNormalization()(conv2_2)
    pool2 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="same")(conv2_2)

    conv3_1 = Conv2D(128, [3, 3], padding="same", activation="relu")(pool2)
    conv3_1 = LayerNormalization()(conv3_1)
    conv3_2 = Conv2D(128, [3, 3], padding="same", activation="relu")(conv3_1)
    conv3_2 = LayerNormalization()(conv3_2)
    pool3 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="same")(conv3_2)

    conv4_1 = Conv2D(256, [3, 3], padding="same", activation="relu")(pool3)
    conv4_1 = LayerNormalization()(conv4_1)
    conv4_2 = Conv2D(256, [3, 3], padding="same", activation="relu")(conv4_1)
    conv4_2 = LayerNormalization()(conv4_2)
    pool4 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="same")(conv4_2)

    conv5_1 = Conv2D(512, [3, 3], padding="same", activation="relu")(pool4)
    conv5_1 = LayerNormalization()(conv5_1)
    conv5_2 = Conv2D(512, [3, 3], padding="same", activation="relu")(conv5_1)
    conv5_2 = LayerNormalization()(conv5_2)
    deconv5 = Conv2DTranspose(256, [4, 4], [2, 2], padding="same", activation="relu")(conv5_2)

    conv6_1 = Conv2D(256, [3, 3], padding="same", activation="relu")(add([deconv5, conv4_2]))
    conv6_1 = LayerNormalization()(conv6_1)
    conv6_2 = Conv2D(256, [3, 3], padding="same", activation="relu")(conv6_1)
    conv6_2 = LayerNormalization()(conv6_2)
    deconv6 = Conv2DTranspose(128, [4, 4], [2, 2], padding="same", activation="relu")(conv6_2)

    conv7_1 = Conv2D(128, [3, 3], padding="same", activation="relu")(add([deconv6, conv3_2]))
    conv7_1 = LayerNormalization()(conv7_1)
    conv7_2 = Conv2D(128, [3, 3], padding="same", activation="relu")(conv7_1)
    conv7_2 = LayerNormalization()(conv7_2)
    deconv7 = Conv2DTranspose(64, [4, 4], [2, 2], padding="same", activation="relu")(conv7_2)

    conv8_1 = Conv2D(64, [3, 3], padding="same", activation="relu")(add([deconv7, conv2_2]))
    conv8_1 = LayerNormalization()(conv8_1)
    conv8_2 = Conv2D(64, [3, 3], padding="same", activation="relu")(conv8_1)
    conv8_2 = LayerNormalization()(conv8_2)
    deconv8 = Conv2DTranspose(32, [4, 4], [2, 2], padding="same", activation="relu")(conv8_2)

    conv9_1 = Conv2D(32, [3, 3], padding="same", activation="relu")(add([deconv8, conv1_2]))
    conv9_1 = LayerNormalization()(conv9_1)
    conv9_2 = Conv2D(32, [3, 3], padding="same", activation="relu")(conv9_1)
    conv9_2 = LayerNormalization()(conv9_2)
    deconv9 = Conv2D(num_classes, [1, 1], padding="same", activation="relu")(conv9_2)
    softmax = Softmax()(deconv9)
    return softmax