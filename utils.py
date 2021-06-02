import numpy as np
import pickle
import cv2
import os
import configparser


def get_env_variable(var, is_boolean_value=False):
    '''
    return the variable from the environment
    '''
    config = configparser.ConfigParser()
    config.read('.env')
    return config['DEFAULT'].getboolean(var) if is_boolean_value else config['DEFAULT'][var]


def download_dataset():
    if len(os.listdir('dataset')) <= 2:  # there is the .gitignore
        # prepare key for kaggle
        os.environ['KAGGLE_USERNAME'] = get_env_variable('KAGGLE_USERNAME')
        os.environ['KAGGLE_KEY'] = get_env_variable('KAGGLE_KEY')

        # download dataset
        os.system('kaggle datasets download -d longnguyen2306/bacteria-detection-with-darkfield-microscopy -p dataset')
        # with zipfile.ZipFile("dataset/bacteria-detection-with-darkfield-microscopy.zip", "r") as zip_ref:
        #     zip_ref.extractall("dataset")
        # zip_ref = zipfile.ZipFile('dataset/bacteria-detection-with-darkfield-microscopy.zip')  # create zipfile object
        # zip_ref.extractall('dataset')  # extract file to dir
        # zip_ref.close()  # close file
        print(os.system('pwd'))
        os.system('unzip -u -qq dataset/bacteria-detection-with-darkfield-microscopy.zip -d dataset')


def create_layer_of_color(mask):
    '''
    The mask image has three color - background, blood cell and bacteria
    For the processing, we need this object as one hot encode in three different layer
    '''
    img = np.asarray(mask)
    tmp = np.zeros((int(get_env_variable('HEIGHT')), int(get_env_variable('WIDTH')), 3), dtype=int)

    tmp[:, :, 0] = img == 0
    tmp[:, :, 1] = img == 1
    tmp[:, :, 2] = img == 2
    return tmp


def create_train_validation_set():
    '''
    Function to save the train and validation set. After that, we can use it for train the models
    '''
    path_imgs = "dataset/images"
    path_masks = "dataset/masks"
    list_imgs = os.listdir(path_imgs)
    list_masks = os.listdir(path_masks)

    imgs = []
    masks = []
    for image_name in list_imgs:
        image = cv2.imread(os.path.join(path_imgs, image_name))
        image = cv2.resize(image, (int(get_env_variable('HEIGHT')), int(get_env_variable('WIDTH'))))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # image = clahe.apply(image)
        imgs.append(image)

        # find the mask and append it
        mask = cv2.imread(os.path.join(path_masks, image_name))
        mask = cv2.resize(mask, (int(get_env_variable('HEIGHT')), int(get_env_variable('WIDTH'))))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # masks.append(np.around(tf.keras.utils.to_categorical(mask, 3)))
        masks.append(create_layer_of_color(mask))

    imgs = np.asarray(imgs)
    masks = np.asarray(masks)

    with open(os.path.join('dataset/set_imgs'), 'wb') as set_imgs:
        pickle.dump(imgs, set_imgs)

    with open(os.path.join('dataset/set_masks'), 'wb') as set_masks:
        pickle.dump(masks, set_masks)
