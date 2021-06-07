import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from augmentation import augment
from utils import create_train_validation_set, download_dataset, get_env_variable
from models_utils import Unet, tversky_loss, mean_IoU, predict_mask_and_plot,  Show_Intermediate_Pred
import os
from tensorflow.keras.models import load_model

# ##############
# Settings
# ##############
WIDTH = int(get_env_variable('WIDTH'))
HEIGHT = int(get_env_variable('HEIGHT'))
NUM_CLASSES = 3
EPOCHS = 20
TRAIN_MODEL = bool(get_env_variable('TRAIN_MODEL', is_boolean_value=True))


# ##############
# download dataset and prepare it
# ##############
download_dataset()
if 'set_imgs' not in os.listdir("dataset") or 'set_masks' not in os.listdir("dataset"):
    create_train_validation_set()

# open the sets
with open('dataset/set_imgs', 'rb') as ts:
    imgs = pickle.load(ts)

with open('dataset/set_masks', 'rb') as ts:
    masks = pickle.load(ts)

# ##############
# create the train (and validation) masks and images
# ##############
how_many_training_sample = 330
train_images = imgs[0:how_many_training_sample, :, :, :]
train_masks = masks[0:how_many_training_sample, :, :, :]
val_images = imgs[how_many_training_sample:, :, :, :]
val_masks = masks[how_many_training_sample:, :, :, :]

print(f"Total images: {len(imgs)}")
print(f"[Split mask and image - Train] => Train  Image{train_images.shape} Train Mask:{train_masks.shape}")
print(f"[Split mask and image - Val  ] => Val Image {val_images.shape} Val Mask:{val_masks.shape}")

# plot some examples
fig, axs = plt.subplots(1, 5)
fig.set_size_inches(20, 6)
axs[0].imshow(train_images[0]), axs[0].set_title('Original Image')
axs[1].imshow(train_masks[0] * 255), axs[1].set_title('Mask [Bg + cell + bact]')
axs[2].imshow(train_masks[0, :, :, 1]), axs[2].set_title('blood cell')
axs[3].imshow(train_masks[0, :, :, 2]), axs[3].set_title('bacteria')
axs[4].imshow(train_masks[0, :, :, 0]), axs[4].set_title('Background')

# ##############
# Data Augmentation
# ##############
# we create two instances with the same arguments
data_gen_args = dict(  # featurewise_center=True,
    # featurewise_std_normalization=True,
    # rotation_range=90.,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # zoom_range=0.2
    horizontal_flip=True, # Randomly flip inputs horizontally.
    vertical_flip=True # Randomly flip inputs vertically.
)
image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args,
                                                                rescale=1.0 / 255.,
                                                                preprocessing_function=augment
                                                                )
mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_generator = image_datagen.flow(
    train_images,
    seed=seed,
    batch_size=8)

mask_generator = mask_datagen.flow(
    train_masks,
    seed=seed,
    batch_size=8)

# combine generators into one which yields image and masks
# train
train_generator = (pair for pair in zip(image_generator, mask_generator))
# validation
generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.)
val_generator = generator.flow(val_images, val_masks)

# x, y = next(train_generator)
# plt.imshow(x[0]), plt.show()
# plt.imshow(y[0]), plt.show()
#
# print(np.unique(y[0]), y[0].shape, np.max(x))
#tf.keras.backend.clear_session()


# ##############
# Model Section
# ##############
# model = get_model((HEIGHT,WIDTH), num_classes)
model = Unet(HEIGHT, WIDTH, NUM_CLASSES)
# model.summary()
# loss_weights = {0: 0.01, 1: 0.5, 2: 0.5}
# w = [[loss_weights[0], loss_weights[1], loss_weights[2]]] * WIDTH
# h = [w] * HEIGHT
# loss_mod = np.array(h)

if not TRAIN_MODEL:
    saved_model = load_model('saved_model/bact_seg_10_epoch_v1.h5', compile=False)
    image_index = 14
    predict_mask_and_plot(val_images[image_index], val_masks[image_index], saved_model)
else:

    #### HERE WE TRAIN THE MODEL
    tf.config.experimental_run_functions_eagerly(True)
    model.compile(optimizer="rmsprop", loss=tversky_loss,
                  metrics=['accuracy', tversky_loss, mean_IoU]
                  # loss_weights=loss_mod
                  )

    callbacks = [
        Show_Intermediate_Pred(val_images[13], val_masks[13])
        # tf.keras.callbacks.ModelCheckpoint("bacteria.h5", save_best_only=True, monitor="val_accuracy"),
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3),
        # tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15)
    ]

    # Train the model, doing validation at the end of each epoch.

    epochs = 30

    # history = model.fit(
    #           x=train_images,
    #           y=train_masks ,
    #           epochs=epochs,
    #           callbacks=callbacks,
    #           validation_data=(val_images, val_masks))

    history = model.fit(
        train_generator,
        epochs=epochs,
        callbacks=callbacks,
        # validation_data=val_generator,
        steps_per_epoch=60
        # class_weight=classes_weights
    )

    acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(acc) + 1)
    #
    # plt.plot(epochs, acc, 'b', color="orange", label='Training acc')
    # plt.plot(epochs, val_acc, 'b', color="green", label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.legend()
    #
    # plt.figure()
    # plt.plot(epochs, loss, 'b', color="orange", label='Training loss')
    # plt.plot(epochs, val_loss, 'b', color="green", label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()
    # plt.show()

    predict_mask_and_plot(val_images[13], val_masks[13], model)

