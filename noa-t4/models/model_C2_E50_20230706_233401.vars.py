# training images should be grayscale and resized already, and ready for training. Predicting images will be grayscale/resize on the fly
IMAGES_TRAIN_DIR = 'data/train-300'

RESIZE_FILL_COLOR = 255 # used in inference, and must be consistent with training image's preprocessing. will cause bad inferencing otherwise.

TRAIN_NUM_EPOCHS = 50
TRAIN_BATCH_SIZE = 50
TRAIN_LEARNING_RATE= 0.001
TRAIN_EVAL_SPLIT = 0.7

RANDOM_SEED = 101
MODEL_NAME_PREFIX = "model"
MODEL_OUT_DIR = 'models'
