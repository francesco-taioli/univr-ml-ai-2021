import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from augmentation import augment
from utils import create_train_validation_set, download_dataset, get_env_variable
from models_utils import mean_IoU, predict_mask_and_plot, Show_Intermediate_Pred, pixel_accuracy, \
    get_lr_metric, pixel_wise_loss
import os
from tensorflow.keras.models import load_model
from datetime import datetime
from pathlib import Path
import numpy as np
from learning_rate_schedulers import CyclicLR, WarmUpLearningRateScheduler
from sklearn.model_selection import train_test_split
from cross_validation import cross_validation


# ##########################################
# Models
# ##########################################
from models.Fcn8 import Fcn8
from models.Seg_Net import Seg_Net
from models.U_Net import Unet
from models.PSP_Net import PSP_Net
from models.Link_Net import Link_Net
from segmentation_models.losses import JaccardLoss, CategoricalCELoss
from segmentation_models.metrics import IOUScore

# python -m tensorboard.main --logdir=S:\train_data\logs --host=127.0.0.1 --port 6006 <--change logdir based on env variable TRAIN_DATA

# ##########################################
# Settings
# ##########################################
NUM_CLASSES = 3
WIDTH = int(get_env_variable('WIDTH'))
HEIGHT = int(get_env_variable('HEIGHT'))
EPOCHS = int(get_env_variable('EPOCHS'))
TRAIN_MODEL = bool(get_env_variable('TRAIN_MODEL', is_boolean_value=True))
SAVED_MODEL = bool(get_env_variable('SAVED_MODEL', is_boolean_value=True))
CROSS_VALIDATION = bool(get_env_variable('CROSS_VALIDATION', is_boolean_value=True))
BATCH_SIZE = int(get_env_variable('BATCH_SIZE'))
BPE = int(get_env_variable('BATCHES_PER_EPOCH'))  # batches per epoch


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
train_images, val_images, train_masks, val_masks = train_test_split(imgs, masks, test_size=0.3, random_state=3)
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
    vertical_flip=True  # Randomly flip inputs vertically.
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
    saved_model_name = 'LinkNet_288_CL_RMS.h5'  # should be stored on TRAIN_DATA/saved_model
    print(f"Start evaluating the model {saved_model_name} ...")
    saved_model = load_model(os.path.join(get_env_variable('TRAIN_DATA'), 'saved_model', saved_model_name),
                             compile=False)
    for index in range(0, 10):
        predict_mask_and_plot(val_images[index], val_masks[index], saved_model, save=True, index=index)
else:
    print("Start training the model...")
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.001)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    lr_metric = get_lr_metric(optimizer)

    loss = JaccardLoss()  # class_weights=[0.1, 5.0, 10.0])
    # loss = CategoricalCELoss(class_weights=[0.1, 5.0, 10.0])
    # loss = pixel_wise_loss()

    # model = Seg_Net((HEIGHT, WIDTH, 3), NUM_CLASSES)
    # model = Unet(HEIGHT, WIDTH, NUM_CLASSES)
    # model = PSP_Net((HEIGHT, WIDTH, 3))
    model = Link_Net((HEIGHT, WIDTH, 3))
    # model = Fcn8((HEIGHT, WIDTH, 3), NUM_CLASSES).get_model()
    model.summary()
    metrics = [mean_IoU, pixel_accuracy, lr_metric]

    # MODEL TRAINING
    tf.config.experimental_run_functions_eagerly(True)
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=metrics
                  )

    Path(os.path.join(get_env_variable('TRAIN_DATA'), 'logs')).mkdir(parents=True, exist_ok=True)
    Tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(get_env_variable('TRAIN_DATA'), 'logs',
                                                                      datetime.now().strftime("%Y%m%d-%H%M%S")),
                                                 write_graph=False
                                                 )
    callbacks = [
        # Show_Intermediate_Pred(val_images[13], val_masks[13]),
        # tf.keras.callbacks.ModelCheckpoint("bacteria.h5", save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(monitor='val_mean_IoU', patience=20, min_delta=0.001,
                                         restore_best_weights=True),
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mean_IoU', factor=0.8, patience=5, min_lr=0.001, mode='auto'),
        CyclicLR(base_lr=0.001, max_lr=0.01, mode='triangular2', step_size=BPE * 5),
        # WarmUpLearningRateScheduler(warmup_batches=BPE * 10, init_lr=0.01, verbose=0, decay_steps=BPE * 40, alpha=0.001),
        # Tensorboard
    ]

    if CROSS_VALIDATION:
        print("Starting cross validation ...")
        all_history = cross_validation(model, imgs, masks, EPOCHS, BPE, image_datagen, mask_datagen, BATCH_SIZE,
                                       metrics, optimizer, loss)
    else:
        # Train the model, doing validation at the end of each epoch.
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            callbacks=callbacks,
            validation_data=(val_images, val_masks),
            steps_per_epoch=BPE
        )

    # dd/mm/YY H:M:S
    if SAVED_MODEL:
        model.save(os.path.join(get_env_variable('TRAIN_DATA'), "saved_model",
                                f"Model-{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.h5"))
    val_images_index = 13
    predict_mask_and_plot(val_images[val_images_index], val_masks[val_images_index], model)
