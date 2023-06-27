import os
import torch
from torch import nn, optim
from torchvision import transforms
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import vars
import util
from data_util import load_and_note_imageFolder, split_and_note_dataset

from NoaT4MultiCnn import NoaT4MultiCnn
from Note import Note

# globals

note = Note("__note_train.txt")
torch.manual_seed(vars.RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

note.addline("########## ########## ##########")
note.addtimeline_()
note.addline_(f"Random seed: {vars.RANDOM_SEED}")
note.addline_(f"Device: {device}")
note.flush()

dataset = load_and_note_imageFolder(root=vars.IMAGES_TRAIN_DIR, 
                                    transform=transforms.Compose([
                                        transforms.RandomVerticalFlip(p=0.5), 
                                        transforms.RandomRotation(degrees=20),
                                        transforms.Resize((vars.IMAGE_HEIGHT, vars.IMAGE_WIDTH)),
                                        transforms.Grayscale(),
                                        transforms.ToTensor()
                                    ]),
                                    note=note)

assert vars.NUM_CLASSES == len(dataset.classes), f"Expecting {vars.NUM_CLASSES} class types, but actual {len(dataset.classes)}"

dataset_train, dataset_eval = split_and_note_dataset(dataset, 
                                                    train_eval_split=vars.TRAIN_EVAL_SPLIT,
                                                    note=note)
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
model_name = f"{vars.MODEL_NAME_PREFIX}_C{vars.NUM_CLASSES}_E{vars.TRAIN_NUM_EPOCHS}_{ts}"
model = NoaT4MultiCnn()
note.addline_(f"Model name: {model_name}")
note.addline_(f"Model: {model}")
note.flush()

writer = SummaryWriter(f"runs/{vars.MODEL_NAME_PREFIX}")

# functions 

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

def train():
    start_time = datetime.now()
    note.addline_(f"Train epochs would be: {vars.TRAIN_NUM_EPOCHS}")
    note.addline_(f"Train start: {start_time}")
    note.flush()

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
            # Mostly: assert pred_logits.size() == torch.Size([vars.TRAIN_BATCH_SIZE, vars.NUM_CLASSES])
            batch_loss = criterion(pred_logits, y)
            loss_per_epoch += batch_loss.item()
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            
        prev_loss = loss_per_epoch
        prev_accuracy = accuracy = eval()

        if epoch_i % 10 == 0:
            note.addline_(f"Epoch {epoch_i}, loss={loss_per_epoch:.2f}, accuracy={accuracy:.2f}").flush()

        writer.add_scalars(model_name, {
            'loss': loss_per_epoch,
            'accuracy': accuracy
        }, epoch_i)

    end_time = datetime.now()
    note.addline_(f"Train end: {end_time}. Took seconds: {(end_time-start_time).total_seconds()}")
    note.addline_(f"Training result: epochs={vars.TRAIN_NUM_EPOCHS}, loss={prev_loss:.2f}, accuracy={prev_accuracy:.2f}")
    note.flush()

def save():
    os.makedirs(vars.MODEL_OUT_DIR, exist_ok=True)
    model_path = f"{vars.MODEL_OUT_DIR}/{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    note.addline_(f"Saved model: {model_path}")
    note.flush()

def on_train_done():
    print("Running on_train_done.sh")
    os.system('./on_train_done.sh')


train()
save()
note.flush()
note.describe()

on_train_done()
