# training images should be grayscale and resized already, and ready for training. Predicting images will be grayscale/resize on the fly
IMAGES_TRAIN_DIR = 'data/train-300'

RESIZE_FILL_COLOR = 255 # used in inference, and must be consistent with training image's preprocessing. will cause bad inferencing otherwise.

TRAIN_NUM_EPOCHS = 100
TRAIN_BATCH_SIZE = 50
TRAIN_LEARNING_RATE= 0.001
TRAIN_EVAL_SPLIT = 0.7

RANDOM_SEED = 101
MODEL_NAME_PREFIX = "model"
MODEL_OUT_DIR = 'models'

#### generated: ###
model_binary_path = 'models/model_C2_E100_20230707_233907.pt'
model_class_archive_path = 'models/model_C2_E100_20230707_233907.py'
classes = ['noa', 't4']
