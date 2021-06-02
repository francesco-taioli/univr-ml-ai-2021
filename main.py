import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from augmentation import augment
from utils import create_train_validation_set, download_dataset
from models_utils import Unet, tversky_loss, mean_IoU
import os

WIDTH = 256
HEIGHT = 256

download_dataset()

if 'set_imgs' not in os.listdir("dataset") or 'set_masks' not in os.listdir("dataset"):
 create_train_validation_set()

#open the sets
with open('dataset/set_imgs', 'rb') as ts:
    imgs = pickle.load(ts)

with open('dataset/set_masks', 'rb') as ts:
    masks = pickle.load(ts)

# create the train (and validation) masks and images
how_many_training_sample = 330
train_images = imgs[0:how_many_training_sample, :, :, :]
train_masks = masks[0:how_many_training_sample, :, :, :]
val_images = imgs[how_many_training_sample :, :, :, :]
val_masks = masks[how_many_training_sample:, :, :, :]

print(f"Total images: {len(imgs)}")
print(f"[Split mask and image - Train] => Train  Image{train_images.shape} Train Mask:{train_masks.shape}")
print(f"[Split mask and image - Val  ] => Val Image {val_images.shape} Val Mask:{val_masks.shape}")

# plot some examples
fig, axs = plt.subplots(1, 5)
fig.set_size_inches(20,6)
axs[0].imshow(train_images[0]), axs[0].set_title('Original Image')
axs[1].imshow(train_masks[0] * 255), axs[1].set_title('Mask [Bg + cell + bact]')
axs[2].imshow(train_masks[0, :, :, 1]), axs[2].set_title('blood cell')
axs[3].imshow(train_masks[0, :, :, 2]), axs[3].set_title('bacteria')
axs[4].imshow(train_masks[0, :, :, 0]), axs[4].set_title('Background')



# we create two instances with the same arguments
data_gen_args = dict(# featurewise_center=True,
                     # featurewise_std_normalization=True,
                     # rotation_range=90.,
                     # width_shift_range=0.1,
                     # height_shift_range=0.1,
                     # zoom_range=0.2
                     horizontal_flip=True,
                     vertical_flip=True
                     )
image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args,
                                                                rescale= 1.0 /255.,
                                                                preprocessing_function=augment
                                                                )
mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
# image_datagen.fit(train_images, augment=True, seed=seed)
# mask_datagen.fit(train_masks, augment=True, seed=seed)

image_generator = image_datagen.flow(
    train_images,
    seed=seed,
    batch_size=8)

mask_generator = mask_datagen.flow(
    train_masks,
    seed=seed,
    batch_size=8)

# combine generators into one which yields image and masks
train_generator = (pair for pair in zip(image_generator, mask_generator))

generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1.0 /255.)
val_generator = generator.flow(val_images, val_masks)

x, y = next(train_generator)
plt.imshow(x[0]), plt.show()
plt.imshow(y[0]), plt.show()

print(np.unique(y[0]), y[0].shape, np.max(x))

tf.keras.backend.clear_session()

# Build model
num_classes = 3
# model = get_model((HEIGHT,WIDTH), num_classes)
model = Unet(HEIGHT, WIDTH, num_classes)
# model.summary()
# loss_weights = {0: 0.01, 1: 0.5, 2: 0.5}
# w = [[loss_weights[0], loss_weights[1], loss_weights[2]]] * WIDTH
# h = [w] * HEIGHT
# loss_mod = np.array(h)


tf.config.experimental_run_functions_eagerly(True)
model.compile(optimizer="rmsprop", loss=tversky_loss,
              metrics=['accuracy', tversky_loss, mean_IoU]
              # loss_weights=loss_mod
              )

callbacks = [
    # tf.keras.callbacks.ModelCheckpoint("bacteria.h5", save_best_only=True, monitor="val_accuracy"),
    # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3),
    # tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15)
]

# Train the model, doing validation at the end of each epoch.

epochs = 20

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

acc = history.history['acc']
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


def predict_mask(index):
  img, mask = val_images[index], val_masks[index]
  image = np.reshape(img / 255., newshape=(1, img.shape[0], img.shape[1], img.shape[2])) #/ 255.

  pred = model.predict(image)

  res = np.argmax(pred[0], axis=-1)
  f = np.zeros((256, 256, 3))
  f[:, :, 0] = res == 0
  f[:, :, 1] = res == 1
  f[:, :, 2] = res == 2

  fig, axs = plt.subplots(1, 3)
  fig.set_size_inches(20,6)
  axs[0].imshow(img), axs[0].set_title('Original Image')
  axs[1].imshow(mask*255), axs[1].set_title('True Mask')
  axs[2].imshow(f), axs[2].set_title('Pred mask')
  plt.show()

predict_mask(13)

