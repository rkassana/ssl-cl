import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from utility.util import *
from torch.utils import data
import random
from data_loader.loader import DataSetName
from collections import Counter


class Dataset_Creator(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y


class Continual_Data_Loader():
    """
    The main class to load a continum of data
    """

    def __init__(self, dataset_list, num_tasks, num_class_in_task, train_batch_size, num_samples,
                 num_workers, random_seed=None, split_fold_train_valid=10):

        self.dataset_list = dataset_list
        self.num_tasks = num_tasks
        self.num_class_in_task = num_class_in_task
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.split_fold_train_valid = split_fold_train_valid

    def _datasets_loader(self):
        """
        Load datassets which is combination of MNIST, CIFAR10 and CIFAR100
        This function modifies targets of datasets to avoid conflicts
        :return:
        mix_dataset : concatination of Tensors of 3 datasets
        dataset_name_label_list : Tuple for dataset and class_number e.g. ('MNIST', 1), ('CIFAR10', 15)
        train_datasets_labels : list of targets
        """

        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

        transform = transforms.Compose(
            [transforms.ToTensor()]
        )

        train_mix_dataset = None
        test_mix_dataset = None
        dataset_name_label_list = []
        train_datasets_labels = []
        test_datasets_labels = []

        ### CIFAR 10
        if DataSetName.CIFAR10 in self.dataset_list:
            cifar10_trainset = torchvision.datasets.CIFAR10(root='../data/', train=True, download=True,
                                                            transform=transform)

            cifar10_testset = torchvision.datasets.CIFAR10(root='../data/', train=False, download=True,
                                                           transform=transform)

            train_datasets_labels = [target + 10 for target in cifar10_trainset.targets]
            test_datasets_labels = [target + 10 for target in cifar10_testset.targets]

            dataset_name_label_list = [('CIFAR10', i) for i in range(10, 20)]

        ### CIFAR 100
        if DataSetName.CIFAR100 in self.dataset_list:
            cifar100_trainset = torchvision.datasets.CIFAR100(root='../data/', train=True, download=True,
                                                              transform=transform)

            cifar100_testset = torchvision.datasets.CIFAR100(root='../data/', train=False, download=True,
                                                             transform=transform)

            train_datasets_labels += [target + 100 for target in cifar100_trainset.targets]
            test_datasets_labels += [target + 100 for target in cifar100_testset.targets]

            dataset_name_label_list += [('CIFAR100', i) for i in range(100, 200)]

            if 'cifar10_trainset' in locals():
                train_mix_dataset = torch.from_numpy(
                    np.concatenate((cifar10_trainset.data, cifar100_trainset.data), axis=0))
                test_mix_dataset = torch.from_numpy(
                    np.concatenate((cifar10_testset.data, cifar100_testset.data), axis=0))
            else:
                train_mix_dataset = torch.from_numpy(cifar10_trainset.data)
                test_mix_dataset = torch.from_numpy(cifar100_testset.data)

        ### MNIST
        if DataSetName.MNIST in self.dataset_list:
            #
            transform = torchvision.transforms.Compose([transforms.Resize(32),
                                                        transforms.ToTensor(),
                                                        transforms.Lambda(lambda x: torch.cat([x, x, x], 0))])

            MNIST_trainset = torchvision.datasets.MNIST(root='../data/', train=True, download=True, transform=transform)
            MNIST_testset = torchvision.datasets.MNIST(root='../data/', train=False, download=True, transform=transform)

            dataset_name_label_list += [('MNIST', i) for i in range(0, 10)]
            train_datasets_labels += MNIST_trainset.targets
            test_datasets_labels += MNIST_testset.targets

            train_mix_dataset = torch.cat((MNIST_trainset.data, train_mix_dataset.data), 0)
            test_mix_dataset = torch.cat((MNIST_testset.data, test_mix_dataset.data), 0)

        return train_mix_dataset, test_mix_dataset, train_datasets_labels, test_datasets_labels, dataset_name_label_list

    def cal_num_sample(self, class_sample_dict, class_indicies):
        """
        In case the number of samples are limited and classes have different number of samples,
        the minimum number of samples will be picked from each class
        :param class_sample_dict: The dictionary of number of samples
        :param class_indicies:
        :return:
        """
        sub_dict = dict((k, class_sample_dict[k]) for k in class_indicies)
        min_sample = min(sub_dict.values())
        return min_sample

    def update_num_sample(self, class_sample_dict, task_class_index, num_removed_samples, classes_list):
        """
        Each time some samples picked the number of remaining will be picked and if all samples of a class
        is picked, it will be removed from the list of availables training classes
        :param class_sample_dict:
        :param task_class_index:
        :param num_removed_samples:
        :param classes_list:
        :return:
        """
        for k in task_class_index:
            if (num_removed_samples == None):
                class_sample_dict[k] = 0
            else:
                class_sample_dict[k] = class_sample_dict[k] - num_removed_samples

            if (class_sample_dict[k] == 0):
                classes_list = [(item, name) for (item, name) in classes_list if name != k]

        return class_sample_dict, classes_list

    def continum(self,
                 params,
                 selection_appraoch=None,
                 random_seed=None):

        """
        This functions create a continum of data for training a continual learning algorithm
        :param num_task: total number of requested tasks
        :param num_class_in_task: number of classes in each task
        :param num_samples: number of samples in each task
        :param selection_appraoch: The way the data should be selected for each class in each task
        ;param params : parameters of data loader
        :param random_seed:
        :return:
        """

        if random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

        tasks = []

        train_set, test_Set, train_set_labels, test_set_labels, classes_list = self._datasets_loader()

        # Create Validation list
        valid_size = len(train_set_labels) // self.split_fold_train_valid

        indices = list(range(0, len(train_set_labels)))
        random.shuffle(indices)

        train_indices = indices[valid_size:]
        # train_labels_indices
        train_labels_indices = [train_set_labels[i] for i in train_indices]

        valid_indices = indices[:valid_size]
        valid_labels_indicies = [train_set_labels[i] for i in valid_indices]

        # num_samples_each_class = [train_set_labels.count(item) for item in set(train_set_labels)]
        train_rem_sample_class_dict = dict(Counter(train_set_labels))

        test_labels_indices = list(range(0, len(test_set_labels)))

        valid_class_index = []
        for counter in range(0, self.num_tasks):
            task_list = random.sample(classes_list, self.num_class_in_task)
            class_index = [item for (a, item) in task_list]
            valid_class_index += class_index

            train_data_idx = [train_indices[i] for i in range(len(train_indices)) if
                              train_labels_indices[i] in class_index]
            valid_data_idx = [valid_indices[i] for i in range(len(valid_indices)) if
                              valid_labels_indicies[i] in valid_class_index]
            test_data_idx = [test_labels_indices[i] for i in range(len(test_labels_indices))
                             if test_set_labels[i] in valid_class_index]

            final_idx_train = []
            final_idx_valid = []
            final_idx_test = []

            if self.num_samples == None:
                # All data for each class
                final_idx_train = train_data_idx
                final_idx_valid = valid_data_idx
                final_idx_test = test_data_idx

                train_rem_sample_class_dict, classes_list = \
                    self.update_num_sample(class_sample_dict=train_rem_sample_class_dict,
                                           task_class_index=class_index,
                                           num_removed_samples=None,
                                           classes_list=classes_list)

            elif selection_appraoch == 'Equal':
                # specific samples of all data and equal number of each class
                number_of_samples_each_class_train = self.num_samples // self.num_class_in_task
                min_samples = self.cal_num_sample(train_rem_sample_class_dict, class_index)
                number_of_samples_each_class_train = min(number_of_samples_each_class_train, min_samples)
                # number_of_samples_each_class_valid = number_of_samples_each_class_train // self.split_fold_train_valid

                for label in class_index:
                    train_index = [j for j in train_data_idx if train_set_labels[j] == label]
                    final_idx_train += random.sample(train_index, number_of_samples_each_class_train)

                for label in valid_class_index:
                    valid_index = [j for j in valid_data_idx if train_set_labels[j] == label]
                    # final_idx_valid += random.sample(valid_index, number_of_samples_each_class_valid // len(valid_class_index))
                    final_idx_valid += valid_index
                    test_index = [j for j in test_data_idx if test_set_labels[j] == label]
                    final_idx_test += test_index

                train_rem_sample_class_dict, classes_list = \
                    self.update_num_sample(class_sample_dict=train_rem_sample_class_dict,
                                           task_class_index=class_index,
                                           classes_list=classes_list,
                                           num_removed_samples=min_samples)

            else:
                # specifc samples of all data
                final_idx_train = random.sample(train_data_idx, self.num_samples)

                for label in valid_class_index:
                    valid_index = [j for j in valid_data_idx if train_set_labels[j] == label]
                    final_idx_valid += valid_index
                    test_index = [j for j in test_data_idx if test_set_labels[j] == label]
                    final_idx_test += test_index

                train_rem_sample_class_dict, classes_list = \
                    self.update_num_sample(class_sample_dict=train_rem_sample_class_dict,
                                           task_class_index=class_index,
                                           classes_list=classes_list,
                                           num_removed_samples=None)

            task_train_ds = torch.utils.data.Subset(train_set, final_idx_train)
            task_train_labels = [train_set[i] for i in final_idx_train]
            training_set = Dataset_Creator(task_train_ds, task_train_labels)
            training_generator = data.DataLoader(training_set, **params)

            task_valid_ds = torch.utils.data.Subset(train_set, final_idx_valid)
            task_valid_labels = [train_set[i] for i in final_idx_valid]
            validation_set = Dataset_Creator(task_valid_ds, task_valid_labels)
            validation_generator = data.DataLoader(validation_set, **params)

            task_test_ds = torch.utils.data.Subset(test_Set, final_idx_test)
            task_test_labels = [test_Set[i] for i in final_idx_test]
            testing_set = Dataset_Creator(task_test_ds, task_test_labels)
            test_generator = data.DataLoader(testing_set, **params)

            tasks.append((training_generator, validation_generator, test_generator))

            ## Remove samples from the training set
            train_indices = [item for item in train_indices if item not in final_idx_train]
            train_labels_indices = [train_set_labels[i] for i in train_indices]
        return tasks


params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 0}

dataset_list = [DataSetName.CIFAR10, DataSetName.CIFAR100]

cdl = Continual_Data_Loader(dataset_list=dataset_list,
                            num_tasks=2,
                            num_class_in_task=3,
                            train_batch_size=32,
                            num_samples=1000,
                            num_workers=0)

tasks = cdl.continum(params=params,
                     selection_appraoch='Equal')

print('yes')