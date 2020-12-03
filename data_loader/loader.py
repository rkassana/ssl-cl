import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from enum import Enum
from utility.util import *

class GenHelper(Dataset):
    def __init__(self, mother, length, mapping):
        self.mapping = mapping
        self.length = length
        self.mother = mother

    def __getitem__(self, index):
        return self.mother[self.mapping[index]]

    def __len__(self):
        return self.length

class Reply_buffer():
    def __init__(self):
        self.x = torch.Tensor()
        self.y = torch.Tensor()
        self.labels = list()

def train_valid_split(ds, split_fold=10, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    dslen = len(ds)
    indices = list(range(dslen))
    valid_size = dslen // split_fold
    np.random.shuffle(indices)
    train_mapping = indices[valid_size:]
    valid_mapping = indices[:valid_size]
    train = GenHelper(ds, dslen - valid_size, train_mapping)
    valid = GenHelper(ds, valid_size, valid_mapping)
    return train, valid


def create_dataset(org_labels, task_lbls, base_set, is_train):
    indices = []
    for index, elem in enumerate(base_set.targets):
        if elem in task_lbls:
            indices.append(index)
    x = base_set.data[indices]
    if len(x.shape) == 4:
        x = x.transpose((0, 3, 1, 2))
    if len(x.shape) == 3:
        x = x.unsqueeze(1)
    # x = x.astype('float')
    y = np.array(base_set.targets)[indices]
    if type(x) != torch.Tensor:
        x = torch.tensor(x)
    tensor_y = torch.tensor(y)
    my_dataset = torch.utils.data.TensorDataset(x, tensor_y)
    my_dataset.classes = [org_labels[item] for item in task_lbls]
    my_dataset.class_to_idx = {org_labels[val]: val for val in task_lbls}
    my_dataset.train = is_train
    return my_dataset

def create_dataset_by_buffer(org_labels, task_lbls, base_set, is_train, reply_buffer):
    indices = []
    for index, elem in enumerate(base_set.targets):
        if elem in task_lbls:
            indices.append(index)
    x = base_set.data[indices]
    if len(x.shape) == 4:
        x = x.transpose((0, 3, 1, 2))
    if len(x.shape) == 3:
        x = x.unsqueeze(1)
    # x = x.astype('float')
    y = np.array(base_set.targets)[indices]
    if type(x) != torch.Tensor:
        x = torch.tensor(x)
    tensor_y = torch.tensor(y)
    if reply_buffer == None :
        my_dataset = torch.utils.data.TensorDataset(x, tensor_y)
        my_dataset.classes = [org_labels[item] for item in task_lbls]
        my_dataset.class_to_idx = {org_labels[val]: val for val in task_lbls}
        reply_buffer = Reply_buffer()

        reply_buffer.x = x[:1000]
        reply_buffer.y = tensor_y[:1000]
        reply_buffer.labels = task_lbls

    else:
        my_dataset = torch.utils.data.TensorDataset(torch.cat((x, reply_buffer.x), 0), torch.cat((tensor_y, reply_buffer.y), 0))
        task_lbls = task_lbls + reply_buffer.labels
        my_dataset.classes = [org_labels[item] for item in task_lbls]
        my_dataset.class_to_idx = {org_labels[val]: val for val in task_lbls}

        reply_buffer.x = torch.cat((reply_buffer.x, x[:1000]), 0)
        reply_buffer.y = torch.cat((reply_buffer.y, tensor_y[:1000]), 0)
        reply_buffer.labels = task_lbls

    my_dataset.train = is_train
    return my_dataset, reply_buffer


def loads(trainset, testset, pre_testset, pre_valset, task_lbls, random_seed, split_fold,
          test_batch_size, train_batch_size, val_batch_size, reply_buffer, num_workers=0):
    if not list(set(task_lbls).intersection(list(range(len(trainset.classes))))).sort() == task_lbls.sort():
        raise Exception(Color.RED.value + "The Selected Labels are Wrong" + Color.END.value)
    labels = trainset.classes
    trainset, reply_buffer = create_dataset_by_buffer(labels, task_lbls, trainset, reply_buffer=reply_buffer, is_train=True)
    testset = create_dataset(labels, task_lbls, testset, is_train=False)
    testset.idx = list(testset.class_to_idx.values())

    if pre_testset is not None:
        testset.idx.extend(pre_testset.idx)
        idx = testset.idx
        testset = torch.utils.data.ConcatDataset([testset, pre_testset])
        testset.idx = idx

    train_set, validation_set = train_valid_split(trainset, split_fold=split_fold, random_seed=random_seed)
    validation_set.idx = list(validation_set.mother.class_to_idx.values())

    if pre_valset is not None:
        validation_set.idx.extend(pre_valset.idx)
        idx = validation_set.idx
        validation_set = torch.utils.data.ConcatDataset([validation_set, pre_valset])
        validation_set.idx = idx

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    validationloader = torch.utils.data.DataLoader(validation_set, batch_size=val_batch_size, shuffle=True,
                                                   num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
    return testloader, trainloader, validationloader, testset, validation_set, reply_buffer


def get_dataloaders(dataset_name, task_lbls, reply_buffer, pre_testset=None, pre_valset=None, split_fold=10, train_batch_size=16,
                    test_batch_size=500,
                    val_batch_size=500, random_seed=32, num_workers=0):

    if dataset_name == DataSetName.CIFAR10:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif dataset_name == DataSetName.CIFAR100:
        transform = transforms.Compose(
            [transforms.ToTensor()])
    elif dataset_name == DataSetName.MNIST:
        transform = transforms.Compose(
            [transforms.ToTensor()])

    if dataset_name == DataSetName.CIFAR10:
        trainset = torchvision.datasets.CIFAR10(root='../data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='../data/', train=False, download=True, transform=transform)
    elif dataset_name == DataSetName.CIFAR100:
        trainset = torchvision.datasets.CIFAR100(root='../data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='../data/', train=False, download=True, transform=transform)
    elif dataset_name == DataSetName.MNIST:
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        trainset = torchvision.datasets.MNIST(root='../data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='../data/', train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,
                                                  num_workers=num_workers)

    testloader, trainloader, validationloader, testset, validation_set, reply_buffer_ = loads(trainset, testset, pre_testset,
                                                                               pre_valset, task_lbls,
                                                                               random_seed, split_fold,
                                                                               test_batch_size,
                                                                               train_batch_size, val_batch_size, reply_buffer=reply_buffer,
                                                                               num_workers=num_workers)
    return trainloader, validationloader, testloader, testset, validation_set, reply_buffer_



class DataSetName(Enum):
    CIFAR10 = (1, 10)
    CIFAR100 = (2, 100)
    MNIST = (3, 10)
    FashionMNIST = (4, 10)


def create_tasks(dataset_name, cil_step, cil_start):
    tasks = []
    all_task = list(range(int(dataset_name.value[1])))
    tasks.append(cil_start)
    for e in cil_start: all_task.remove(e)
    while len(all_task) > 0:
        ls = all_task[:cil_step]
        for i in range(cil_step):
            if len(all_task) > 0:
                del all_task[0]
            else:
                break
        tasks.append(ls)
    return tasks


def get_dataset_name(arg):
    if arg.lower() == 'cifar10':
        return DataSetName.CIFAR10
    elif arg.lower() == 'cifar100':
        return DataSetName.CIFAR100
    elif arg.lower() == 'mnist':
        return DataSetName.MNIST
    elif arg.lower() == 'fmnist':
        return DataSetName.FashionMNIST


