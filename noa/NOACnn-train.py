import torch
import os
import torchvision
from torch import nn, optim
from torchvision import transforms
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import Vars as Vars
from NOACnn import NOACnn

def timestamp():
    """like 20230613_001228"""
    import datetime 
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

torch.manual_seed(101)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

train_ratio = 0.7
image_size = 300
batch_size = 40
learning_rate= 0.001
num_epochs = 50

assert os.path.isdir(Vars.IMAGES_TRAIN_DIR), f"Training data folder not found: {Vars.IMAGES_TRAIN_DIR}"

train_transforms=transforms.Compose([
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20, fill=255), # want white background
    transforms.Resize(image_size),
    transforms.Grayscale(),
    transforms.ToTensor()
    ])

dataset = torchvision.datasets.ImageFolder(root=Vars.IMAGES_TRAIN_DIR, 
                                           transform=train_transforms)

num_train = int(len(dataset) * train_ratio)
num_eval = len(dataset) - num_train

train_ds, eval_ds = torch.utils.data.random_split(dataset, [num_train, num_eval])

# print data statistics 

print("=== Total ===")
print(f"Total classes: {dataset.classes}")
print(f"Total datapoints of {dataset.classes[0]}: {dataset.targets.count(0)}")
print(f"Total datapoints of {dataset.classes[1]}: {dataset.targets.count(1)}")

print("\n=== Train ===")
print(f"Train datapoints: {len(train_ds)}")

num_class0, num_class1 = 0, 0
for img, label_index in iter(train_ds):
    if label_index == 0:
        num_class0 += 1
    else:
        num_class1 += 1
print(f"In train portion, class {dataset.classes[0]} has {num_class0}")
print(f"                  class {dataset.classes[1]} has {num_class1}")

print("\n=== Eval ===")
num_class0, num_class1 = 0, 0
for img, label_index in iter(eval_ds):
    if label_index == 0:
        num_class0 += 1
    else:
        num_class1 += 1
print(f"In eval portion, class {dataset.classes[0]} has {num_class0}")
print(f"                 class {dataset.classes[1]} has {num_class1}")

# batch loader
train_loader = torch.utils.data.DataLoader(train_ds,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last=False)

eval_loader = torch.utils.data.DataLoader(eval_ds,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          drop_last=False)

# model
ts = timestamp()
model_name_suffix = f"{ts}_epochs{num_epochs}"
model = NOACnn()
print(f"Model: {model}")

writer = SummaryWriter("runs/NOACnn")

# functions 

def eval():
    model.eval()
    num_correct = 0

    with torch.no_grad():
        for X, y in eval_loader:
            if device.type != 'cpu':
                X, y = X.to(device), y.to(device)
                model.to(device)
            y_hat = model(X).flatten()
            y_hat_labels = torch.sigmoid(y_hat) > 0.5
            num_correct += (y_hat_labels == y).type(torch.int32).sum().item()
    
    accuracy = num_correct / (len(eval_loader.dataset))
    return accuracy

def train():
    start_time = datetime.now()
    print(f"Start time: {start_time}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    prev_loss = None

    for epoch_i in range(num_epochs):
        loss_per_epoch = 0.0

        for batch_i, (X, y) in enumerate(train_loader):
            if device.type != 'gpu':
                X, y = X.to(device), y.to(device)
                model.to(device)

            optimizer.zero_grad()
            y_hat = model(X).flatten()
            loss = criterion(y_hat, y.type(torch.float32))
            loss.backward()
            optimizer.step()

            prev_loss = loss.item()
            loss_per_epoch += loss.item()

            if epoch_i % 100 == 0:
                print(f"Epoch {epoch_i}, batch #{batch_i}, loss={loss.item()}")

        writer.add_scalars(model_name_suffix, {
            'loss': loss_per_epoch,
            'accuracy': eval()
        }, epoch_i)

    end_time = datetime.now()
    print(f"End time: {end_time}. Took seconds: {(end_time-start_time).total_seconds()}")
    print(f"Training result: \n\tdevice={device}, \n\tepochs={num_epochs}, \n\tloss={prev_loss}")

def save():
    os.makedirs(Vars.MODEL_OUT_DIR, exist_ok=True)
    model_path = f"{Vars.MODEL_OUT_DIR}/{type(model).__name__}_{model_name_suffix}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model: {model_path}")


# start
train()
save()


'''
End time: 2023-06-17 11:29:49.656695. Took seconds: 75.302899
Training result: 
        device=cuda, 
        epochs=50, 
        loss=0.00029796946910209954
Saved model: models/NOACnn_20230617_112834_epochs50.pt
'''