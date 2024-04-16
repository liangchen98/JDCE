import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.array(ind).T

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def cluster_evaluation(y_true, y_pred):
    nmi = normalized_mutual_info_score(y_true, y_pred)
    acc = cluster_acc(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    return nmi, acc, ari


def load_cifar10(Batch_size):
    transform_cifar10 = transforms.Compose([
        # transforms.Resize([96, 96]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_cifar10)
    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_cifar10)
    dataset_cifar10 = data.ConcatDataset([cifar10_train, cifar10_test])
    dataloader_cifar10 = DataLoader(dataset=dataset_cifar10, batch_size=Batch_size, shuffle=True, num_workers=12,
                                    pin_memory=False)

    total_data_num = len(dataset_cifar10)
    classes = len(cifar10_train.classes)

    return dataloader_cifar10, total_data_num, classes


def load_stl10(Batch_size):
    transform_stl10 = transforms.Compose([
        # transforms.Resize([96, 96]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    stl10_train = torchvision.datasets.STL10(root='./data', split='train', download=False, transform=transform_stl10)
    stl10_test = torchvision.datasets.STL10(root='./data', split='test', download=False, transform=transform_stl10)
    dataset_stl10 = stl10_test + stl10_train
    dataloader_stl10 = DataLoader(dataset=dataset_stl10, batch_size=Batch_size, shuffle=True, num_workers=12,
                                  pin_memory=False)

    total_data_num = len(dataset_stl10)
    classes = len(stl10_train.classes)

    return dataloader_stl10, total_data_num, classes


def load_cifar100(Batch_size):
    transform_cifar100 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ])
    cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=False,
                                                   transform=transform_cifar100)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=False,
                                                  transform=transform_cifar100)
    dataset_cifar100 = cifar100_train + cifar100_test
    dataloader_cifar100 = DataLoader(dataset=dataset_cifar100, batch_size=Batch_size, shuffle=True, num_workers=12,
                                     pin_memory=False)
    total_data_num = len(dataset_cifar100)
    classes = 20

    return dataloader_cifar100, total_data_num, classes


def load_imagenet10(Batch_size):
    a = np.load('./data/imagenet10/data.npy')
    b = np.load('./data/imagenet10/label.npy')
    tensor_data = torch.zeros(size=[13000, 3, 96, 96])
    for i in range(13000):
        img = Image.fromarray(a[i])
        transform = transforms.Compose([
            # transforms.Resize([96, 96]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        tensor_data[i] = transform(img)
    tensor_label = torch.from_numpy(b)
    dataset = TensorDataset(tensor_data, tensor_label)
    class_num = 10
    data_loader = DataLoader(dataset, shuffle=True, batch_size=Batch_size, num_workers=8)
    total_data_num = 13000
    return data_loader, total_data_num, class_num


def load_imagedog(Batch_size):
    a = np.load('./data/imagenet_dog/data.npy')
    b = np.load('./data/imagenet_dog/label.npy')
    tensor_data = torch.zeros(size=[19500, 3, 96, 96])
    for i in range(19500):
        img = Image.fromarray(a[i])
        transform = transforms.Compose([
            # transforms.Resize([96, 96]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        tensor_data[i] = transform(img)
    tensor_label = torch.from_numpy(b)
    dataset = TensorDataset(tensor_data, tensor_label)
    class_num = 15
    data_loader = DataLoader(dataset, shuffle=True, batch_size=Batch_size, num_workers=8)
    total_data_num = 19500
    return data_loader, total_data_num, class_num


def cifar100_to_cifar20(target):
    """
    CIFAR100 to CIFAR 20 dictionary.
    This function is from IIC GitHub.
    """

    class_dict = {0: 4,
                  1: 1,
                  2: 14,
                  3: 8,
                  4: 0,
                  5: 6,
                  6: 7,
                  7: 7,
                  8: 18,
                  9: 3,
                  10: 3,
                  11: 14,
                  12: 9,
                  13: 18,
                  14: 7,
                  15: 11,
                  16: 3,
                  17: 9,
                  18: 7,
                  19: 11,
                  20: 6,
                  21: 11,
                  22: 5,
                  23: 10,
                  24: 7,
                  25: 6,
                  26: 13,
                  27: 15,
                  28: 3,
                  29: 15,
                  30: 0,
                  31: 11,
                  32: 1,
                  33: 10,
                  34: 12,
                  35: 14,
                  36: 16,
                  37: 9,
                  38: 11,
                  39: 5,
                  40: 5,
                  41: 19,
                  42: 8,
                  43: 8,
                  44: 15,
                  45: 13,
                  46: 14,
                  47: 17,
                  48: 18,
                  49: 10,
                  50: 16,
                  51: 4,
                  52: 17,
                  53: 4,
                  54: 2,
                  55: 0,
                  56: 17,
                  57: 4,
                  58: 18,
                  59: 17,
                  60: 10,
                  61: 3,
                  62: 2,
                  63: 12,
                  64: 12,
                  65: 16,
                  66: 12,
                  67: 1,
                  68: 9,
                  69: 19,
                  70: 2,
                  71: 10,
                  72: 0,
                  73: 1,
                  74: 16,
                  75: 12,
                  76: 9,
                  77: 13,
                  78: 15,
                  79: 13,
                  80: 16,
                  81: 19,
                  82: 2,
                  83: 4,
                  84: 6,
                  85: 19,
                  86: 5,
                  87: 5,
                  88: 8,
                  89: 19,
                  90: 18,
                  91: 1,
                  92: 2,
                  93: 15,
                  94: 6,
                  95: 0,
                  96: 17,
                  97: 8,
                  98: 14,
                  99: 13}

    return class_dict[target]


class JDCEDatasets(torch.utils.data.Dataset):
    def __init__(self, img_tensor, ensemble_tensor):
        self.img = img_tensor
        self.ensemble = ensemble_tensor

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        x = self.img[idx]
        y = self.ensemble[idx]

        return x, y
