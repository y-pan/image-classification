
NUM_CLASSES = 2 # must be consistent with folder

IMAGE_CHANNELS = 1
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300

# training images should be grayscale and resized already, and ready for training. Predicting images will be grayscale/resize on the fly
IMAGES_TRAIN_DIR = '/home/yun/Documents/code/static/noa-t4-multi/train'
MODEL_OUT_DIR = 'models'

RANDOM_SEED = 101

MODEL_NAME_PREFIX = "NoaT4MultiCnn"

TRAIN_BATCH_SIZE = 40


TRAIN_EVAL_SPLIT = 0.7
TRAIN_LEARNING_RATE= 0.001
TRAIN_NUM_EPOCHS = 50

RESIZE_FILL_COLOR = 255