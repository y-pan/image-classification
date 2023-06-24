import os
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import Vars
from NoaT4MultiCnn import NoaT4MultiCnn
from data_helper import load_and_note_imageFolder, split_and_note_dataset
from Note import Note

def timestamp():
    """like 20230613_001228"""
    import datetime 
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

note = Note("__note_train.txt")
torch.manual_seed(Vars.RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

note.addline("########## ########## ##########")
note.addtimeline_()
note.addline_(f"Random seed: {Vars.RANDOM_SEED}")
note.addline_(f"Device: {device}")
note.flush()

dataset = load_and_note_imageFolder(root=Vars.IMAGES_TRAIN_DIR, 
                                    transform=transforms.Compose([
                                        transforms.RandomVerticalFlip(p=0.5), 
                                        transforms.RandomRotation(degrees=20),
                                        transforms.Resize((Vars.IMAGE_HEIGHT, Vars.IMAGE_WIDTH)),
                                        transforms.Grayscale(),
                                        transforms.ToTensor()
                                    ]),
                                    note=note)

assert Vars.NUM_CLASSES == len(dataset.classes), f"Expecting {Vars.NUM_CLASSES} class types, but actual {len(dataset.classes)}"

dataset_train, dataset_eval = split_and_note_dataset(dataset, 
                                                    train_eval_split=Vars.TRAIN_EVAL_SPLIT,
                                                    note=note)
# batch loader
loader_train = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=Vars.TRAIN_BATCH_SIZE,
                                           shuffle=True,
                                           drop_last=False)

loader_eval = torch.utils.data.DataLoader(dataset_eval,
                                          batch_size=Vars.TRAIN_BATCH_SIZE,
                                          shuffle=False,
                                          drop_last=False)

# model
ts = timestamp()
model_name = f"{Vars.MODEL_NAME_PREFIX}_C{Vars.NUM_CLASSES}_E{Vars.TRAIN_NUM_EPOCHS}_{ts}"
model = NoaT4MultiCnn()
note.addline_(f"Model name: {model_name}")
note.addline_(f"Model: {model}")
note.flush()

writer = SummaryWriter(f"runs/{Vars.MODEL_NAME_PREFIX}")

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
    note.addline_(f"Train epochs would be: {Vars.TRAIN_NUM_EPOCHS}")
    note.addline_(f"Train start: {start_time}")
    note.flush()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Vars.TRAIN_LEARNING_RATE, betas=(0.9, 0.999))
    prev_loss = None

    for epoch_i in range(Vars.TRAIN_NUM_EPOCHS):
        loss_per_epoch = 0.0

        for batch_i, (X, y) in enumerate(loader_train):
            if device.type != 'gpu':
                X, y = X.to(device), y.to(device)
                model.to(device)

            pred_logits = model(X) 
            # Mostly: assert pred_logits.size() == torch.Size([Vars.TRAIN_BATCH_SIZE, Vars.NUM_CLASSES])
            batch_loss = criterion(pred_logits, y)
            loss_per_epoch += batch_loss.item()
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            
        prev_loss = loss_per_epoch
        accuracy = eval()

        if epoch_i % 10 == 0:
            note.addline_(f"Epoch {epoch_i}, loss={loss_per_epoch}, accuracy={accuracy}").flush()

        writer.add_scalars(model_name, {
            'loss': loss_per_epoch,
            'accuracy': accuracy
        }, epoch_i)

    end_time = datetime.now()
    note.addline_(f"Train end: {end_time}. Took seconds: {(end_time-start_time).total_seconds()}")
    note.addline_(f"Training result: epochs={Vars.TRAIN_NUM_EPOCHS}, loss={prev_loss}")
    note.flush()

def save():
    os.makedirs(Vars.MODEL_OUT_DIR, exist_ok=True)
    model_path = f"{Vars.MODEL_OUT_DIR}/{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    note.addline_(f"Saved model: {model_path}")
    note.flush()


train()
save()
note.flush()
print("Done!")