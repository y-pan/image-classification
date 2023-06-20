import os
import pprint
import torch
import torchvision
from torch import nn
from PIL import Image

import Vars
from NOACnn import NOACnn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = torchvision.datasets.ImageFolder(root=Vars.IMAGES_TRAIN_DIR).classes
print(f"classes: {classes}")


# test
def predictOne(imgPath, expectingClassName=None):
    imgPIL = loadAndResizeImgGrayScale(
        imgPath, sizeTupleOfWidthHeight=(300, 300), paddingColor=255)
    imgTensor = torchvision.transforms.functional.to_tensor(
        imgPIL).unsqueeze(0)

    # print(f"test image tensor shape: {imgTensor.shape}")
    assert imgTensor.shape == torch.Size([1, 1, 300, 300])

    with torch.no_grad():
        model.eval()
        pred = model(imgTensor)
        predClass = int(torch.sigmoid(pred) > 0.5)
        predClassName = classes[predClass]
        print(f"pred: {pred.data}")
        print(f"predClass: {predClass}")
        print(f"predClassName: {predClassName}")
        if expectingClassName:
            print(f"expectingClassName: {expectingClassName}")
            print(f"=== OK? {expectingClassName == predClassName}")
        return predClass


def loadAndResizeImgGrayScale(imgSrcPath, sizeTupleOfWidthHeight=(300, 300), paddingColor=255):
    '''
    Fit the specified weight/height, padded. 
    paddingColor: 0 -> 255
    Output: image data resized
    '''

    imgSrc = Image.open(imgSrcPath).convert('L')
    # print(imgSrc.size) # original size (1498, 966)
    # imgSrc.thumbnail(sizeTupleOfWidthHeight, Image.Resampling.LANCZOS) # worked on Mac, but not Unbutu somehow
    # worked on ubuntu, not tested on Mac
    imgSrc.thumbnail(sizeTupleOfWidthHeight, Image.LANCZOS)

    # print(imgSrc.size) # new size (500, 322)
    w, h = imgSrc.size
    w_out, h_out = sizeTupleOfWidthHeight
    wdif = w_out - w
    hdif = h_out - h
    newImg = Image.new("L", (w_out, h_out), paddingColor)
    newImg.paste(imgSrc, (wdif//2, hdif//2))
    return newImg


def testOne(model, imgPath, expectingClassName):
    # imgPath="data/t4-classification/test/t4/Screenshot 2023-05-30 at 00.13.40__0-100.png"
    imgPIL = loadAndResizeImgGrayScale(
        imgPath, sizeTupleOfWidthHeight=(300, 300), paddingColor=255)
    imgTensor = torchvision.transforms.functional.to_tensor(
        imgPIL).unsqueeze(0)

    # print(f"test image tensor shape: {imgTensor.shape}")
    assert imgTensor.shape == torch.Size([1, 1, 300, 300])

    with torch.no_grad():
        model.eval()
        pred = model(imgTensor)
        predClass = int(torch.sigmoid(pred) > 0.5)
        predClassName = classes[predClass]
        # print(f"pred: {pred.data}")
        # print(f"predClass: {predClass}")
        # print(f"predClassName: {predClassName}")
        return predClassName, expectingClassName == predClassName



def testAllUnderFolder(model, srcDir, expectingClassName):
    correctCount, incorrectCount = 0, 0
    incurrectImages = []
    for curPath, _folder, files in os.walk(srcDir):
        for f in files:
            imgFullPath = os.path.join(curPath, f)
            pred, is_correct = testOne(model, imgFullPath, expectingClassName)
            if is_correct:
                correctCount += 1
            else:
                incorrectCount += 1
                incurrectImages.append(imgFullPath)
    print("")
    print(f"srcDir: {srcDir}")
    print(f"expectingClassName: {expectingClassName}")
    print(f"accuracy: {correctCount}/{correctCount+incorrectCount}")
    print(f"incorrect predictions:")
    pprint.pprint(incurrectImages)


if __name__ == '__main__':
    model = NOACnn()
    modelPath = f"{Vars.MODEL_OUT_DIR}/NOACnn_20230617_112834_epochs50.pt"
    model.load_state_dict(torch.load(modelPath, map_location=device))

    # testAllUnderFolder(
    #     model,
    #     "/home/yun/Documents/code/static/noa/random/noa_yes", expectingClassName="noa_yes")
    
    testAllUnderFolder(
        model,
        "/home/yun/Documents/code/static/t4/train/not_t4", expectingClassName="noa_no")

'''
classes: ['noa_no', 'noa_yes']

srcDir: /home/yun/Documents/code/static/noa/random/noa_yes
expectingClassName: noa_yes
accuracy: 4/4
incorrect predictions:
[]
'''