from sklearn.model_selection import KFold
import numpy as np

def cross_validation(model, images, masks, EPOCHS, callbacks, BPE, image_datagen, mask_datagen, BATCH_SIZE):
    all_history = {}
    fold = 1
    kfold = KFold(n_splits=5, shuffle=True )
    for train, val in kfold.split(images, masks):
        image_generator = image_datagen.flow(images[train], seed=1, batch_size=BATCH_SIZE)
        mask_generator = mask_datagen.flow(masks[train], seed=1, batch_size=BATCH_SIZE)
        train_generator = (pair for pair in zip(image_generator, mask_generator))

        val_images = images[val] / 255.
        val_images = val_images.astype(np.float32)
        val_masks = masks[val].astype(np.float32)

        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            callbacks=callbacks,
            validation_data=(val_images, val_masks),
            steps_per_epoch=BPE
        )
        all_history[f"fold-{fold}"] = history
        fold += 1