import torch
import torchvision
from datetime import datetime
import Vars
from Note import Note


randomseed = Vars.RANDOM_SEED
# num_classes_expected = Vars.NUM_CLASSES

torch.manual_seed(randomseed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dataset_describe(dataset, class_list, note):
    note.addline(f"Total size: {len(dataset)}")
    note.addline(f"Classes: {class_list}")
    note.addline(f"Number of classes: {len(class_list)}")

    for i in range(len(class_list)):
        count = dataset.targets.count(i)
        percentage = count * 100 / len(dataset)
        note.addline(f"Size of '{class_list[i]}': {count} | {percentage:.2f}%")
    note.flush()

def dataset_split_describe(dataset_split, class_list, note):
    count_per_class_train = [0] * len(class_list)

    for img, label_index in iter(dataset_split):
        count_per_class_train[label_index] += 1

    for i in range(len(class_list)):
        note.addline_(f"Size of '{class_list[i]}': {count_per_class_train[i]} | {count_per_class_train[i]/len(dataset_split) * 100:.2f}%")

def load_and_note_imageFolder(root, 
                              transform=None,
                              note=Note()):
    
    dataset = torchvision.datasets.ImageFolder(root=root, 
                                               transform=transform)
    note.addline(f"Image folder: {root}")
    dataset_describe(dataset=dataset, class_list=dataset.classes, note=note)

    note.flush()
    note.describe()
    return dataset

def split_and_note_dataset(dataset, train_eval_split=0.7, note=Note()):
    num_train = int(len(dataset) * train_eval_split)
    num_eval = len(dataset) - num_train

    train_ds, eval_ds = torch.utils.data.random_split(dataset, [num_train, num_eval])
    note.addline_(f"Train-eval split factor: {train_eval_split}")
    note.addline_(f"Train datapoints: {len(train_ds)}")
    note.addline_(f"Eval datapoints: {len(eval_ds)}")

    note.addline_("=== Train Split ===")
    dataset_split_describe(dataset_split=train_ds, class_list=dataset.classes, note=note)

    note.addline_("=== Eval Split ===")
    dataset_split_describe(dataset_split=eval_ds, class_list=dataset.classes, note=note)

    note.flush()
    # count_per_class_train = [0] * len(dataset.classes)

    # for img, label_index in iter(train_ds):
    #     count_per_class_train[label_index] += 1

    # for i in range(len(dataset.classes)):
    #     note.addline_(f"Size of '{dataset.classes[i]}': {count_per_class_train[i]} | {count_per_class_train[i]/len(train_ds) * 100:.2f}%")

    # note.addline_("=== Eval Split ===")
    # count_per_class_eval = [0] * len(dataset.classes)

    # for img, label_index in iter(eval_ds):
    #     count_per_class_eval[label_index] += 1

    # for i in range(len(dataset.classes)):
    #     note.addline_(f"Size of '{dataset.classes[i]}': {count_per_class_eval[i]} | {count_per_class_eval[i]/len(eval_ds) * 100:.2f}%")

    return train_ds, eval_ds

if __name__ == '__main__':
    note = Note()
    full_ds = load_and_note_imageFolder(root=Vars.IMAGES_TRAIN_DIR, note=note)
    train_ds, eval_ds = split_and_note_dataset(full_ds, train_eval_split=0.7, note=note)
    note.flush()