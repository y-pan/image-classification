# training images should be grayscale and resized already, and ready for training. Predicting images will be grayscale/resize on the fly
IMAGES_TRAIN_DIR = '/home/yun/Documents/code/static/noa-t4-multi/train'

# must be consistent with training images, will fail otherwise 
NUM_CLASSES = 5
IMAGE_CHANNELS = 1
IMAGE_HEIGHT = 300 
IMAGE_WIDTH = 300

RESIZE_FILL_COLOR = 255 # used in inference, and must be consistent with training image's preprocessing. will cause bad inferencing otherwise.

TRAIN_NUM_EPOCHS = 50
TRAIN_BATCH_SIZE = 50
TRAIN_LEARNING_RATE= 0.001
TRAIN_EVAL_SPLIT = 0.7

RANDOM_SEED = 101
MODEL_NAME_PREFIX = "NoaT4MultiCnn"
MODEL_OUT_DIR = 'models'
