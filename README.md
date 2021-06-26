# univr_ml-ai_2021

# before running
- cp .env.example .env
- put your kaggle api settings inside the variable

# Tensorboard
python -m tensorboard.main --logdir=S:\train_data\logs --host=127.0.0.1 --port 6006 <--change logdir based on env variable TRAIN_DATA


# package
 - kaggle
 - albumentations
 - segmentation model - pip install -U segmentation-models