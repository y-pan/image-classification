import os
import pprint
import torch
import torchvision
from torch import nn
from PIL import Image

import vars
from Logger import Logger

from models.model_C2_E100_20230707_205742 import Model, NUM_CLASSES, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH
modelPath = "models/model_C2_E100_20230707_205742.pt"

def image_load_resize_grayscale(img_path_src, 
                                height = IMAGE_HEIGHT,
                                width = IMAGE_WIDTH,
                                padcolor=vars.RESIZE_FILL_COLOR):
    imgSrc = Image.open(img_path_src).convert('L')
    imgSrc.thumbnail((width, height), Image.LANCZOS) # Image.Resampling.LANCZOS

    w, h = imgSrc.size   # new size like (300, 234), and we want (300, 300), then paste it onto (300, 300) white canvas
    wdif = width - w
    hdif = height - h
    white_canvas = Image.new("L", (width, height), padcolor)
    white_canvas.paste(imgSrc, (wdif//2, hdif//2))
    return white_canvas


def image_predict(model, img_path, classes):
    img_pil = image_load_resize_grayscale(
        img_path, 
        height = IMAGE_HEIGHT,
        width = IMAGE_WIDTH,
        padcolor=vars.RESIZE_FILL_COLOR)
    
    img_tensor = torchvision.transforms.functional.to_tensor(
        img_pil).reshape((1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))

    # dim0 is batch size, which is 1 in this case
    assert img_tensor.size()[1:] == torch.Size([IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH])

    with torch.no_grad():
        model.eval()
        pred_logits = model(img_tensor)
        # dim0 is batch size, which is 1 in this case
        assert pred_logits.size()[1:] == torch.Size([NUM_CLASSES])
        value, index = torch.max(pred_logits, 1)
        assert value.size() == index.size() == torch.Size([1])
        return index[0], classes[index[0]]

def image_folder_predict_expected(model, 
                                  dir_src, 
                                  expected_classname_for_all, 
                                  classes,
                                  logger):
    logger.addline_(f"predict images: dir={dir_src}, expected={expected_classname_for_all}")
    corrects, incorrects = 0, 0
    images_incorrect = []
    for curpath, _folder, files in os.walk(dir_src):
        for f in files:
            fullpath = os.path.join(curpath, f)
            pred_index, pred_name = image_predict(model, fullpath, classes=classes)
            if expected_classname_for_all == pred_name:
                corrects += 1
            else:
                incorrects += 1
                images_incorrect.append({"file": f"{fullpath} ", "prediction": pred_name, "truth": expected_classname_for_all})
    logger.newline_()
    logger.addline_(f"srcDir: {dir_src}")
    logger.addline_(f"expectingClassName: {expected_classname_for_all}")
    accuracy = corrects / (corrects+incorrects)
    logger.addline_(f"accuracy: {accuracy:.2f} ({corrects} out of {corrects+incorrects})")
    if images_incorrect:
        logger.addline_(f"incorrect predictions:")
        logger.addlines_(images_incorrect)
    logger.flush()

def self_name():
    return ".".join(os.path.basename(__file__).split(".")[0:-1])

if __name__ == '__main__':

    logger = Logger(f"_{self_name()}.log")
    logger.hr()
    logger.addtimeline_()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # classes = torchvision.datasets.ImageFolder(root=vars.IMAGES_TRAIN_DIR).classes
    classes = ['noa', 't4']

    assert NUM_CLASSES == len(classes)

    logger.addline_(f"classes: {classes}")
    logger.addline_(f"device: {device}")

    model = Model()

    model.load_state_dict(torch.load(modelPath, map_location=device))

    logger.addline_(f"modelPath: {modelPath}")


    # Ubu
    # noa_dirs = ["/home/yun/Documents/code/static/sensitive/noa", 
    #             "/home/yun/Documents/code/static/noa-t4-multi/raw/raw_noa",
    #             "/home/yun/Documents/code/static/noa/train/noa_yes",
    #             ]
    
    # t4_dirs = ["/home/yun/Documents/code/static/sensitive/t4",
    #            "/home/yun/Documents/code/static/t4/train/t4",
    #            "/home/yun/Documents/code/static/noa-t4-multi/raw/raw_t4"]

    # mac
    noa_dirs = [
        # "/Users/yunkuipan/Documents/x/static/noa/random/noa_yes",
        "/home/yun/Documents/code/static/sensitive/noa"
    ]
    
    t4_dirs = [
        # "/Users/yunkuipan/Documents/x/static/t4/train/t4"
        "/home/yun/Documents/code/static/sensitive/t4"
    ]

    for _dir in noa_dirs:
        image_folder_predict_expected(
            model,
            _dir, 
            expected_classname_for_all="noa",
            classes=classes,
            logger=logger)
        
    for _dir in t4_dirs:
        image_folder_predict_expected(
            model,
            _dir, 
            expected_classname_for_all="t4",
            classes=classes,
            logger=logger)
        
    logger.flush()
