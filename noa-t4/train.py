import os
import shutil
import torch
from torch import nn, optim
from torchvision import transforms
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import vars
import util
from data_util import load_image_folder, split_dataset

from Model import Model, NUM_CLASSES, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH
from Logger import Logger

def eval():
    model.eval()
    num_correct = 0

    with torch.no_grad():
        for X, y in loader_eval:
            if device.type != 'cpu':
                X, y = X.to(device), y.to(device)
                model.to(device)
            pred_logits = model(X)
            maxItemValues, maxItemIndices = torch.max(pred_logits, 1)
            num_correct += (maxItemIndices == y).type(torch.int32).sum().item()
    
    accuracy = num_correct / (len(loader_eval.dataset))
    return accuracy

def train(logger):
    start_time = datetime.now()
    logger.addline_(f"Train epochs would be: {vars.TRAIN_NUM_EPOCHS}")
    logger.addline_(f"Train start: {start_time}")
    logger.flush()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=vars.TRAIN_LEARNING_RATE, betas=(0.9, 0.999))
    prev_loss = None
    prev_accuracy = None

    for epoch_i in range(vars.TRAIN_NUM_EPOCHS):
        loss_per_epoch = 0.0

        for batch_i, (X, y) in enumerate(loader_train):
            if device.type != 'gpu':
                X, y = X.to(device), y.to(device)
                model.to(device)

            pred_logits = model(X) 
            # Mostly: assert pred_logits.size() == torch.Size([vars.TRAIN_BATCH_SIZE, NUM_CLASSES])
            batch_loss = criterion(pred_logits, y)
            loss_per_epoch += batch_loss.item()
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            
        prev_loss = loss_per_epoch
        prev_accuracy = accuracy = eval()

        if epoch_i % 10 == 0:
            logger.addline_(f"Epoch {epoch_i}, loss={loss_per_epoch:.2f}, accuracy={accuracy:.2f}")
            logger.flush()

        writer.add_scalars(model_name, {
            'loss': loss_per_epoch,
            'accuracy': accuracy
        }, epoch_i)

    end_time = datetime.now()
    logger.addline_(f"Train end: {end_time}. Took seconds: {(end_time-start_time).total_seconds()}")
    logger.addline_(f"Training result: epochs={vars.TRAIN_NUM_EPOCHS}, loss={prev_loss:.2f}, accuracy={prev_accuracy:.2f}")
    logger.flush()

def save(logger):
    os.makedirs(vars.MODEL_OUT_DIR, exist_ok=True)
    
    # save model instance
    model_path = f"{vars.MODEL_OUT_DIR}/{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    logger.addline_(f"Saved model: {model_path}")

    # save model class code
    model_class_archive = f"{vars.MODEL_OUT_DIR}/{model_name}.py"
    shutil.copy("Model.py", model_class_archive)
    logger.addline_(f"Archived model class: {model_class_archive}")

    # save vars
    vars_archive = f"{vars.MODEL_OUT_DIR}/{model_name}.vars.py"
    shutil.copy("vars.py", vars_archive)
    logger.addline_(f"Archived vars: {vars_archive}")

    logger.flush()
    return model_name, model_path

def on_train_done():
    if not os.path.exists("on_train_done.sh.off.__ignore__"):
        script = './on_train_done.sh'
        print(f"Running {script}")
        os.system(script)
    print("Done!")

def self_name():
    return ".".join(os.path.basename(__file__).split(".")[0:-1])

if __name__ == '__main__':
    # globals
    logger = Logger(f"{self_name()}.log")
    torch.manual_seed(vars.RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.addline("########## ########## ##########")
    logger.addtimeline_()
    logger.addline_(f"Random seed: {vars.RANDOM_SEED}")
    logger.addline_(f"Device: {device}")
    logger.flush()

    dataset = load_image_folder(root=vars.IMAGES_TRAIN_DIR, 
                                        transform=transforms.Compose([
                                            transforms.RandomVerticalFlip(p=0.5), 
                                            transforms.RandomRotation(degrees=20),
                                            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
                                            transforms.Grayscale(),
                                            transforms.ToTensor()
                                        ]),
                                        logger=logger)

    assert NUM_CLASSES == len(dataset.classes), f"Expecting {NUM_CLASSES} class types, but actual {len(dataset.classes)}"

    dataset_train, dataset_eval = split_dataset(dataset, 
                                                        train_eval_split=vars.TRAIN_EVAL_SPLIT,
                                                        logger=logger)
    # batch loader
    loader_train = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=vars.TRAIN_BATCH_SIZE,
                                            shuffle=True,
                                            drop_last=False)

    loader_eval = torch.utils.data.DataLoader(dataset_eval,
                                            batch_size=vars.TRAIN_BATCH_SIZE,
                                            shuffle=False,
                                            drop_last=False)

    # model
    ts = util.timestamp()
    model_name = f"{vars.MODEL_NAME_PREFIX}_C{NUM_CLASSES}_E{vars.TRAIN_NUM_EPOCHS}_{ts}"
    model = Model()
    logger.addline_(f"Model name: {model_name}")
    logger.addline_(f"Model: {model}")
    logger.flush()

    writer = SummaryWriter(f"runs/{vars.MODEL_NAME_PREFIX}")

    train(logger=logger)
    model_name, model_path = save(logger=logger)
    
    logger.flush()
    logger.describe()

    on_train_done()
