# Univr Ml & Ai 

# Setup
- cp .env.example .env (use to setup some variable)
- put your kaggle api settings inside .env (KAGGLE_USERNAME and KAGGLE_KEY)

# Tensorboard
python -m tensorboard.main --logdir=S:\train_data\logs --host=127.0.0.1 --port 6006 
// change logdir based on env variable TRAIN_DATA

# TODO 
- COLAB LINK
# package
 - kaggle
 - albumentations
 - segmentation model - pip install -U segmentation-models
