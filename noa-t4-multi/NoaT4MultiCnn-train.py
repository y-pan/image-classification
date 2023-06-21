import os
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import Vars
from NoaT4MultiCnn import NoaT4MultiCnn

def timestamp():
    """like 20230613_001228"""
    import datetime 
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

torch.manual_seed(Vars.RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

train_transforms=transforms.Compose([
    transforms.RandomVerticalFlip(p=0.5), 
    transforms.RandomRotation(degrees=20),
    transforms.Resize((Vars.IMAGE_HEIGHT, Vars.IMAGE_WIDTH)),
    transforms.Grayscale(),
    transforms.ToTensor()
    ])

dataset = torchvision.datasets.ImageFolder(root=Vars.IMAGES_TRAIN_DIR, 
                                           transform=train_transforms)

num_train = int(len(dataset) * Vars.TRAIN_EVAL_SPLIT)
num_eval = len(dataset) - num_train

train_ds, eval_ds = torch.utils.data.random_split(dataset, [num_train, num_eval])

# print data statistics 

print("=== Total ===")
assert Vars.NUM_CLASSES == len(dataset.classes), \
    f"Vars.NUM_CLASSES, doesn't match actual classes, in image folder: Vars.NUM_CLASSES={Vars.NUM_CLASSES}, actual folders={len(dataset.classes)}, dir={Vars.IMAGES_TRAIN_DIR}"

print(f"Total classes: {dataset.classes}")
for i in range(len(dataset.classes)):
    print(f"Total datapoints of {dataset.classes[i]}: {dataset.targets.count(i)}")

print("\n=== Train Split ===")
print(f"Train datapoints: {len(train_ds)}")

count_per_class_train = [0] * len(dataset.classes)

for img, label_index in iter(train_ds):
    count_per_class_train[label_index] += 1

for i in range(len(dataset.classes)):
    print(f"class name {dataset.classes[i]} has {count_per_class_train[i]}")


print("\n=== Eval Split ===")
count_per_class_eval = [0] * len(dataset.classes)

for img, label_index in iter(eval_ds):
    count_per_class_eval[label_index] += 1

for i in range(len(dataset.classes)):
    print(f"class name {dataset.classes[i]} has {count_per_class_train[i]}")


# batch loader
train_loader = torch.utils.data.DataLoader(train_ds,
                                           batch_size=Vars.TRAIN_BATCH_SIZE,
                                           shuffle=True,
                                           drop_last=False)

eval_loader = torch.utils.data.DataLoader(eval_ds,
                                           batch_size=Vars.TRAIN_BATCH_SIZE,
                                          shuffle=False,
                                          drop_last=False)

# model
ts = timestamp()
model_name_suffix = f"{ts}_epochs{Vars.TRAIN_NUM_EPOCHS}"
model = NoaT4MultiCnn()
print(f"Model: {model}")

writer = SummaryWriter(f"runs/{Vars.MODEL_NAME}")

# functions 

def eval():
    model.eval()
    num_correct = 0

    with torch.no_grad():
        for X, y in eval_loader:
            if device.type != 'cpu':
                X, y = X.to(device), y.to(device)
                model.to(device)
            pred_logits = model(X)
            maxItemValues, maxItemIndices = torch.max(pred_logits, 1)
            num_correct += (maxItemIndices == y).type(torch.int32).sum().item()
    
    accuracy = num_correct / (len(eval_loader.dataset))
    return accuracy

def train():
    start_time = datetime.now()
    print(f"Start time: {start_time}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Vars.TRAIN_LEARNING_RATE, betas=(0.9, 0.999))
    prev_loss = None

    for epoch_i in range(Vars.TRAIN_NUM_EPOCHS):
        loss_per_epoch = 0.0

        for batch_i, (X, y) in enumerate(train_loader):
            if device.type != 'gpu':
                X, y = X.to(device), y.to(device)
                model.to(device)

            pred_logits = model(X) 
            # Mostly: assert pred_logits.size() == torch.Size([Vars.TRAIN_BATCH_SIZE, Vars.NUM_CLASSES])
            loss = criterion(pred_logits, y)
            
            optimizer.zero_grad()
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
    print(f"Training result: \n\tdevice={device}, \n\tepochs={Vars.TRAIN_NUM_EPOCHS}, \n\tloss={prev_loss}")

def save():
    os.makedirs(Vars.MODEL_OUT_DIR, exist_ok=True)
    model_path = f"{Vars.MODEL_OUT_DIR}/{Vars.MODEL_NAME}_{model_name_suffix}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model: {model_path}")


# start
train()
save()

"""
End time: 2023-06-20 00:49:00.745865. Took seconds: 879.547475
Training result: 
        device=cuda, 
        epochs=500, 
        loss=1.3815774764225353e-05
Saved model: models/NoaT4MultiCnn_20230620_003421_epochs500.pt

"""