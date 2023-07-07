import os
import pprint
import torch
import torchvision
from torch import nn
from PIL import Image

from NoaT4MultiCnn import NoaT4MultiCnn
import vars
from Note import Note


def image_load_resize_grayscale(img_path_src, 
                                height = vars.IMAGE_HEIGHT,
                                width = vars.IMAGE_WIDTH,
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
        height = vars.IMAGE_HEIGHT,
        width = vars.IMAGE_WIDTH,
        padcolor=vars.RESIZE_FILL_COLOR)
    
    img_tensor = torchvision.transforms.functional.to_tensor(
        img_pil).reshape((1, vars.IMAGE_CHANNELS, vars.IMAGE_HEIGHT, vars.IMAGE_WIDTH))

    # dim0 is batch size, which is 1 in this case
    assert img_tensor.size()[1:] == torch.Size([vars.IMAGE_CHANNELS, vars.IMAGE_HEIGHT, vars.IMAGE_WIDTH])

    with torch.no_grad():
        model.eval()
        pred_logits = model(img_tensor)
        # dim0 is batch size, which is 1 in this case
        assert pred_logits.size()[1:] == torch.Size([vars.NUM_CLASSES])
        value, index = torch.max(pred_logits, 1)
        assert value.size() == index.size() == torch.Size([1])
        return index[0], classes[index[0]]

def image_folder_predict_expected(model, 
                                  dir_src, 
                                  expected_classname_for_all, 
                                  classes,
                                  note):
    note.addline_(f"predict images: dir={dir_src}, expected={expected_classname_for_all}")
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
    note.newline_()
    note.addline_(f"srcDir: {dir_src}")
    note.addline_(f"expectingClassName: {expected_classname_for_all}")
    accuracy = corrects / (corrects+incorrects)
    note.addline_(f"accuracy: {accuracy:.2f} ({corrects} out of {corrects+incorrects})")
    if images_incorrect:
        note.addline_(f"incorrect predictions:")
        note.addlines_(images_incorrect)
    note.flush()

if __name__ == '__main__':

    note = Note("__note_pred__ignore__.log")
    note.addtimeline_()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = torchvision.datasets.ImageFolder(root=vars.IMAGES_TRAIN_DIR).classes
    assert vars.NUM_CLASSES == len(classes)

    note.addline_(f"classes: {classes}")
    note.addline_(f"device: {device}")

    model = NoaT4MultiCnn()
    # modelPath = "models/NoaT4MultiCnn_20230620_003421_epochs500.pt"
    # modelPath = "models/NoaT4MultiCnn_20230621_220422_epochs50.pt"
    # modelPath = "models/NoaT4MultiCnn_C5_E50_20230624_152322.pt"
    # modelPath = "models/NoaT4MultiCnn_C5_E50_20230626_112316.pt"
    modelPath = "models/NoaT4MultiCnn_C5_E100_20230705_184128.pt"
    model.load_state_dict(torch.load(modelPath, map_location=device))

    note.addline_(f"modelPath: {modelPath}")

    # image_folder_predict_expected(
    #     model,
    #     "/home/yun/Documents/code/static/noa-t4-multi/train/t4", expected_classname_for_all="t4")

    # image_folder_predict_expected(
    #     model,
    #     "/home/yun/Documents/code/static/noa-t4-multi/train/noa", expected_classname_for_all="noa")

    noa_dirs = ["/home/yun/Documents/code/static/sensitive/noa"]
    t4_dirs = ["/home/yun/Documents/code/static/sensitive/t4"]

    for _dir in noa_dirs:
        image_folder_predict_expected(
            model,
            _dir, 
            expected_classname_for_all="noa",
            classes=classes,
            note=note)
        
    for _dir in t4_dirs:
        image_folder_predict_expected(
            model,
            _dir, 
            expected_classname_for_all="t4",
            classes=classes,
            note=note)
        
    note.flush()