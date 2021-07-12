from pathlib import Path
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K
import tensorflow as tf
import os
from learning_rate_schedulers import CyclicLR
from utils import get_env_variable
import numpy as np


def cross_validation(model, images, masks, EPOCHS, BPE, image_datagen, mask_datagen, BATCH_SIZE, metrics, optimizer,
                     loss):
    all_history = {}
    fold = 1
    kfold = KFold(n_splits=3, shuffle=True)  # split images and masks obtaining n_splits
    Path(os.path.join(get_env_variable('TRAIN_DATA'), 'logs', 'folds')).mkdir(parents=True, exist_ok=True)
    mean_meanIoU = 0.0

    model.save_weights("temp.h5")

    model.compile(optimizer=optimizer, loss=loss,
                  metrics=metrics
                  )

    for train, val in kfold.split(images, masks):
        model.load_weights("temp.h5")

        image_generator = image_datagen.flow(images[train], seed=1, batch_size=BATCH_SIZE)
        mask_generator = mask_datagen.flow(masks[train], seed=1, batch_size=BATCH_SIZE)
        train_generator = (pair for pair in zip(image_generator, mask_generator))

        val_images = images[val] / 255.
        val_images = val_images.astype(np.float32)
        val_masks = masks[val].astype(np.float32)

        callbacks = [
            # Show_Intermediate_Pred(val_images[13], val_masks[13]),
            # tf.keras.callbacks.ModelCheckpoint("bacteria.h5", save_best_only=True, monitor="val_accuracy"),
            tf.keras.callbacks.EarlyStopping(monitor='pixel_accuracy', patience=20, min_delta=0.001,
                                             restore_best_weights=True),
            # tf.keras.callbacks.ReduceLROnPlateau(monitor='pixel_accuracy', factor=0.8, patience=5, min_lr=0.001, mode='auto'),
            CyclicLR(base_lr=0.001, max_lr=0.01, mode='triangular2', step_size=BPE * 5),
            # WarmUpLearningRateScheduler(warmup_batches=BPE * 10, init_lr=0.01, verbose=0, decay_steps=BPE * 40, alpha=0.001),
            tf.keras.callbacks.TensorBoard(log_dir=os.path.join(get_env_variable('TRAIN_DATA'), 'logs', 'folds',
                                                                f"fold-{fold}"), write_graph=False)
        ]

        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            callbacks=callbacks,
            validation_data=(val_images, val_masks),
            steps_per_epoch=BPE
        )
        all_history[f"fold-{fold}"] = history
        K.clear_session()
        mean_meanIoU += history.history['val_mean_IoU'][-1]
        fold += 1

    mean_meanIoU /= 3.0
    file_writer = tf.summary.create_file_writer('S:\\train_data\\logs\\folds\\metrics')
    file_writer.set_as_default()
    tf.summary.scalar('cross_val_meanIoU', data=mean_meanIoU, step=1)

    return all_history
