import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from augmentation import augment
from utils import create_train_validation_set, download_dataset, get_env_variable
from models_utils import tversky_loss, mean_IoU, predict_mask_and_plot, Show_Intermediate_Pred, pixel_accuracy, \
    get_lr_metric, weighted_categorical_crossentropy, pixel_wise_loss
from tensorflow.keras.layers import Input
import os
from tensorflow.keras.models import load_model
from datetime import datetime
from pathlib import Path
import numpy as np
# models
from models.Fcn8 import Fcn8
from models.SegNet import SegNet
from models.U_Net import Unet
from learning_rate_schedulers import CyclicLR, WarmUpLearningRateScheduler
from sklearn.model_selection import train_test_split
# python -m tensorboard.main --logdir=S:\train_data\logs --host=127.0.0.1 --port 6006 <--change logdir based on env variable TRAIN_DATA

# ##########################################
# Settings
# ##########################################
WIDTH = int(get_env_variable('WIDTH'))
HEIGHT = int(get_env_variable('HEIGHT'))
NUM_CLASSES = 3
EPOCHS = 30
TRAIN_MODEL = bool(get_env_variable('TRAIN_MODEL', is_boolean_value=True))
SAVED_MODEL = bool(get_env_variable('SAVED_MODEL', is_boolean_value=True))
BATCH_SIZE = 8
# batches per epoch
BPE = int(get_env_variable('BATCHES_PER_EPOCH'))

# ##########################################
# download dataset and prepare it
# ##########################################
download_dataset()
if 'set_imgs' not in os.listdir("dataset") or 'set_masks' not in os.listdir("dataset"):
    create_train_validation_set()

# open the sets
with open('dataset/set_imgs', 'rb') as ts:
    imgs = pickle.load(ts)

with open('dataset/set_masks', 'rb') as ts:
    masks = pickle.load(ts)

# ##########################################
# create the train (and validation) masks and images
# ##########################################
train_images, val_images, train_masks, val_masks = train_test_split(imgs, masks, test_size=0.3)
# normalize the val images. The same operation is performed on train_images in the train generator
val_images = val_images / 255.
val_images = val_images.astype(np.float32)
val_masks = val_masks.astype(np.float32)

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
plt.show()
# ##########################################
# Data Augmentation
# ##########################################
# we create two instances with the same arguments

# here we use ImageDataGenerator with the data_gen_args only for flip vertically and horizontally the
# image and mask. For the complete augmentation, see augment function (preprocessing one)
data_gen_args = dict(
    horizontal_flip=True,  # Randomly flip inputs horizontally.
    vertical_flip=True     # Randomly flip inputs vertically.
    # width_shift_range = 0.2,
    # height_shift_range = 0.2
)
image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args,
                                                                rescale=1.0 / 255.,
                                                                preprocessing_function=augment)
mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
# in this way we are sure that both image and mask are vertically or horizontally flip together
seed = 1
image_generator = image_datagen.flow(train_images, seed=seed, batch_size=BATCH_SIZE)
mask_generator = mask_datagen.flow(train_masks, seed=seed, batch_size=BATCH_SIZE)

# combine generators into one which yields image and masks
# train
train_generator = (pair for pair in zip(image_generator, mask_generator))

# ##########################################
# Model Section
# ##########################################
if not TRAIN_MODEL:
    saved_model = load_model('saved_model/bact_seg_10_epoch_v1.h5', compile=False)
    image_index = 14
    predict_mask_and_plot(val_images[image_index], val_masks[image_index], saved_model)
else:
    # optimizer = tf.keras.optimizers.Adam()
    # optimizer = tf.keras.optimizers.SGD()
    optimizer = tf.keras.optimizers.RMSprop()
    lr_metric = get_lr_metric(optimizer)
    # loss = tversky_loss()
    # loss = weighted_categorical_crossentropy()
    loss = pixel_wise_loss()

    model = SegNet((HEIGHT, WIDTH, NUM_CLASSES), NUM_CLASSES)
    # model = get_model((HEIGHT,WIDTH), num_classes)
    # model = Unet(HEIGHT, WIDTH, NUM_CLASSES)
    # model = Fcn8((HEIGHT, WIDTH, NUM_CLASSES), NUM_CLASSES).get_model()
    # model.summary()

    #### HERE WE TRAIN THE MODEL
    tf.config.experimental_run_functions_eagerly(True)
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=[tversky_loss, mean_IoU, pixel_accuracy, lr_metric]
                  )

    Path(os.path.join(get_env_variable('TRAIN_DATA'), 'logs')).mkdir(parents=True, exist_ok=True)
    Tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(get_env_variable('TRAIN_DATA'), 'logs',
                                                                      datetime.now().strftime("%Y%m%d-%H%M%S")),
                                                 write_graph=False
                                                 )
    callbacks = [
        Show_Intermediate_Pred(val_images[13], val_masks[13])
        # tf.keras.callbacks.ModelCheckpoint("bacteria.h5", save_best_only=True, monitor="val_accuracy"),
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3),
        # tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15),
        # CyclicLR(base_lr=0.001, max_lr=0.01, mode='triangular2', step_size= BPE * 5),
        # WarmUpLearningRateScheduler(warmup_batches=BPE * 10, init_lr=0.01, verbose=0, decay_steps=BPE * 20, alpha=0.001),
        # Tensorboard
    ]

    # Train the model, doing validation at the end of each epoch.
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=(val_images, val_masks),
        # validation_steps=1,
        steps_per_epoch=BPE
    )

    # dd/mm/YY H:M:S
    if SAVED_MODEL:
        model.save(os.path.join("saved_model", f"Model-{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.h5"))
    predict_mask_and_plot(val_images[13], val_masks[13], model)
