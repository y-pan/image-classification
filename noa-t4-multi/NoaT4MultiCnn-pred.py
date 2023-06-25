import os
import pprint
import torch
import torchvision
from torch import nn
from PIL import Image

from NoaT4MultiCnn import NoaT4MultiCnn
import Vars

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = torchvision.datasets.ImageFolder(root=Vars.IMAGES_TRAIN_DIR).classes
print(f"classes: {classes}")

assert Vars.NUM_CLASSES == len(classes)


def image_load_resize_grayscale(img_path_src, 
                                height = Vars.IMAGE_HEIGHT,
                                width = Vars.IMAGE_WIDTH,
                                padcolor=Vars.RESIZE_FILL_COLOR):
    imgSrc = Image.open(img_path_src).convert('L')
    imgSrc.thumbnail((width, height), Image.LANCZOS) # Image.Resampling.LANCZOS

    w, h = imgSrc.size   # new size like (300, 234), and we want (300, 300), then paste it onto (300, 300) white canvas
    wdif = width - w
    hdif = height - h
    white_canvas = Image.new("L", (width, height), padcolor)
    white_canvas.paste(imgSrc, (wdif//2, hdif//2))
    return white_canvas


def image_predict(model, img_path):
    img_pil = image_load_resize_grayscale(
        img_path, 
        height = Vars.IMAGE_HEIGHT,
        width = Vars.IMAGE_WIDTH,
        padcolor=Vars.RESIZE_FILL_COLOR)
    
    img_tensor = torchvision.transforms.functional.to_tensor(
        img_pil).reshape((1, Vars.IMAGE_CHANNELS, Vars.IMAGE_HEIGHT, Vars.IMAGE_WIDTH))

    # dim0 is batch size, which is 1 in this case
    assert img_tensor.size()[1:] == torch.Size([Vars.IMAGE_CHANNELS, Vars.IMAGE_HEIGHT, Vars.IMAGE_WIDTH])

    with torch.no_grad():
        model.eval()
        pred_logits = model(img_tensor)
        # dim0 is batch size, which is 1 in this case
        assert pred_logits.size()[1:] == torch.Size([Vars.NUM_CLASSES])
        value, index = torch.max(pred_logits, 1)
        assert value.size() == index.size() == torch.Size([1])
        return index[0], classes[index[0]]

def image_folder_predict_expected(model, dir_src, expected_classname_for_all):
    corrects, incorrects = 0, 0
    images_incorrect = []
    for curpath, _folder, files in os.walk(dir_src):
        for f in files:
            fullpath = os.path.join(curpath, f)
            pred_index, pred_name = image_predict(model, fullpath)
            if expected_classname_for_all == pred_name:
                corrects += 1
            else:
                incorrects += 1
                images_incorrect.append({"file": fullpath, "prediction": pred_name, "truth": expected_classname_for_all})
    print("")
    print(f"srcDir: {dir_src}")
    print(f"expectingClassName: {expected_classname_for_all}")
    accuracy = corrects / (corrects+incorrects)
    print(f"accuracy: {accuracy:.2f} ({corrects} out of {corrects+incorrects})")
    print(f"incorrect predictions:")
    pprint.pprint(images_incorrect)


if __name__ == '__main__':
    model = NoaT4MultiCnn()
    # modelPath = "models/NoaT4MultiCnn_20230620_003421_epochs500.pt"
    # modelPath = "models/NoaT4MultiCnn_20230621_220422_epochs50.pt"
    modelPath = "models/NoaT4MultiCnn_C5_E50_20230624_152322.pt"
    model.load_state_dict(torch.load(modelPath, map_location=device))

    image_folder_predict_expected(
        model,
        "/home/yun/Documents/code/static/noa-t4-multi/train/t4", expected_classname_for_all="t4")

    image_folder_predict_expected(
        model,
        "/home/yun/Documents/code/static/noa-t4-multi/train/noa", expected_classname_for_all="noa")
