import torch
import torchvision
from datetime import datetime
import Vars
from Note import Note


randomseed = Vars.RANDOM_SEED
# num_classes_expected = Vars.NUM_CLASSES

torch.manual_seed(randomseed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def conv_out_hw(input_height, input_width, kernel_height, kernel_width, padding, stride):
    """
    conv_out_hw(300, 500, 3, 5, 0, 1) => (496, 298)
    conv_out_hw(300, 500, 3, 5, 1, 1) => (498, 300)
    """
    out_height = (input_height - kernel_height + 2 * padding)//stride + 1
    out_width = (input_width - kernel_width + 2 * padding)//stride + 1
    return out_height, out_width

def conv_pool_out_hw(input_height, input_width, kernel_height, kernel_width, padding, stride, pool_height, pool_width):
    """
    conv_out_hw(300, 500, 3, 5, 0, 1) => (496, 298)
    conv_out_hw(300, 500, 3, 5, 1, 1) => (498, 300)
    """
    out_height , out_width= conv_out_hw(input_height, input_width, kernel_height, kernel_width, padding, stride)
    return out_height//pool_height, out_width//pool_width

def __test_dim():
    batch_size, num_channels = 10, 3
    img_height, img_width = 300, 400
    X = torch.randn((batch_size, num_channels, img_height, img_width))

    assert X.size() == torch.Size([batch_size, num_channels, img_height, img_width])
    kernel_height, kernel_width = 4, 5
    padding=1
    stride=1
   
    out_height_expected, out_width_expected = conv_out_hw(input_height=img_height, 
                input_width=img_width, 
                kernel_height=kernel_height, 
                kernel_width=kernel_width, 
                padding=padding, 
                stride=stride)
    conv = torch.nn.Conv2d(num_channels, 8, kernel_size=(kernel_height, kernel_width), padding=padding, stride=stride)
    out_conv = conv(X)
    # print("size1:", out_conv.size())
    # print("h1:", out_height_expected)
    # print("w1:", out_width_expected)
    assert out_conv.size() == torch.Size((batch_size, 8, out_height_expected, out_width_expected))

    # width pooling
    pool_height, pool_width = 7, 8
    out_height_expected2, out_width_expected2 = conv_pool_out_hw(input_height=img_height, 
                input_width=img_width, 
                kernel_height=kernel_height, 
                kernel_width=kernel_width, 
                padding=padding, 
                stride=stride,
                pool_height=pool_height,
                pool_width=pool_width)

    pool = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width))
    out_pool = pool(out_conv)
    assert out_pool.size() == torch.Size([batch_size, 8, out_height_expected2, out_width_expected2])
    # print("A: ", out_pool.size())
    # print("B: ", torch.Size([batch_size, 8, out_height_expected2, out_width_expected2]))


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
    return train_ds, eval_ds

def __test_note():
    note = Note()
    full_ds = load_and_note_imageFolder(root=Vars.IMAGES_TRAIN_DIR, note=note)
    train_ds, eval_ds = split_and_note_dataset(full_ds, train_eval_split=0.7, note=note)
    note.flush()

if __name__ == '__main__':
    __test_dim()
    __test_note()