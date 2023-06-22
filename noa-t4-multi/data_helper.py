import torch
import torchvision
from datetime import datetime
import Vars
from Note import Note


randomseed = Vars.RANDOM_SEED
# num_classes_expected = Vars.NUM_CLASSES

torch.manual_seed(randomseed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_note_imageFolder(root, 
                              transform,
                              note=Note()):
    
    dataset = torchvision.datasets.ImageFolder(root=root, 
                                               transform=transform)
    note.addline(f"Image folder: {root}")
    note.addline(f"Total size: {len(dataset)}")
    note.addline(f"Classes: {dataset.classes}")
    note.addline(f"Number of classes: {len(dataset.classes)}")

    for i in range(len(dataset.classes)):
        count = dataset.targets.count(i)
        percentage = count * 100 / len(dataset)
        note.addline(f"Size of '{dataset.classes[i]}': {count} | {percentage:.2f}%")
    
    note.newline()

    note.flush()
    note.describe()
    return dataset

def split_and_note_dataset(dataset, train_eval_split=0.7, note=Note()):
    num_train = int(len(dataset) * train_eval_split)
    num_eval = len(dataset) - num_train

    train_ds, eval_ds = torch.utils.data.random_split(dataset, [num_train, num_eval])

    note.addline_("\n=== Train Split ===")
    note.addline_(f"Train datapoints: {len(train_ds)}")

    count_per_class_train = [0] * len(dataset.classes)

    for img, label_index in iter(train_ds):
        count_per_class_train[label_index] += 1

    for i in range(len(dataset.classes)):
        note.addline_(f"class name {dataset.classes[i]} has {count_per_class_train[i]}")

    note.addline_("\n=== Eval Split ===")
    count_per_class_eval = [0] * len(dataset.classes)

    for img, label_index in iter(eval_ds):
        count_per_class_eval[label_index] += 1

    for i in range(len(dataset.classes)):
        note.addline_(f"class name {dataset.classes[i]} has {count_per_class_train[i]}")

    return train_ds, eval_ds

if __name__ == '__main__':
    load_and_note_imageFolder(root=Vars.IMAGES_TRAIN_DIR)