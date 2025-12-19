import os

class Config:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
    MODEL_PATH = os.path.join(CHECKPOINT_DIR, f'model-lr.pkl')
    LOG_PATH = os.path.join(CHECKPOINT_DIR, 'model-log.log')

    TRAIN_PATH = os.path.join("data", "vietnam_housing_dataset.csv")