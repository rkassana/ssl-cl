import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import pathlib
import models.simple_model
from data_loader.loader import get_dataloaders
from utility.util import DataSetName
from utility.learning_utils import *
from byol_pytorch import BYOL
import wandb


def freeze_convs(soft_model, sig_model):
    freeze(soft_model)
    freeze(sig_model)
    return soft_model, sig_model


def freeze(model):
    for module in model.layer1:
        freeze_module(module)
    for module in model.layer2:
        freeze_module(module)
    for module in model.layer3:
        freeze_module(module)
    for module in model.layer4:
        freeze_module(module)


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def get_summary(dist):
    summary = []
    for key, val in dist.items():
        avg = np.mean(np.array(val), axis=0)
        summary.append((key, avg))
    return summary


def get_models(dataset_name):
    model = None
    if dataset_name.name == DataSetName.CIFAR100.name:
        model = models.simple_model.ConvModel(100, in_chanel=3)
    elif dataset_name.name == DataSetName.CIFAR10.name:
        model = models.simple_model.ConvModel(10, in_chanel=3)
    elif dataset_name.name == DataSetName.MNIST.name or dataset_name.name == DataSetName.FashionMNIST.name:
        model = models.simple_model.ConvModel(10, in_chanel=1)

    model.name = 'soft_model'
    return model


def perform_cil_tasks(tasks, model, dataset_name, epochs=2, lr=0.0001, report_step=20, approach='dll'):
    pre_testset, pre_valset = None, None
    is_first_task = True
    test_stat, validation_stat, running_losses = list(), list(), list()
    classes_avg = dict()
    for task in tasks:
        print(f"{Color.RED.value}task: {task}{Color.END.value}")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            num_workers = 2
        else:
            num_workers = 0

        trainloader, validationloader, testloader, pre_testset, pre_valset = get_dataloaders(dataset_name=dataset_name,
                                                                                       pre_testset=pre_testset,
                                                                                       pre_valset=pre_valset,
                                                                                       task_lbls=task, num_workers=num_workers)

        if type(model) == type(None):
            model = get_models(dataset_name=dataset_name)

        if approach == 'dll':
            test_stat, validation_stat, running_losses, model = train(model, trainloader, validationloader,
                                                                      testloader,task, test_stat, validation_stat,
                                                                      running_losses, epochs=epochs, lr=lr,
                                                                      report_step=report_step)

        if approach == 'dll':
            preds, y_true, (acc, f1, p) = predict(model, testloader, task)
            print(f'acc:{acc:.4f}, f1:{f1:.4f}, p:{p:.4f}')
            # plot_confusion_matrix(y_true, preds, classes=task, task=task)

            # plt.plot(range(1, len(running_losses) + 1), running_losses, alpha=.6)
            # plt.xlabel("steps")
            # plt.ylabel("loss")
            # plt.title("Running Loss")
            # plt.show()

    plt.plot(range(1, len(validation_stat) + 1), validation_stat, alpha=.6)
    plt.xlabel("steps")
    plt.ylabel("accuracy")
    plt.title("Validation Accuracy")
    plt.show()

    plt.plot(range(1, len(running_losses) + 1), running_losses, alpha=.6)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("Test Accuracy")
    plt.show()


    if approach == 'dll':
        running_losses = np.array(running_losses)
    validation_stat = np.array(validation_stat)
    test_stat = np.array(test_stat)

    path = os.path.join(f'../results/batch_cil/{dataset_name}')
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    if approach == 'dll':
        running_losses_path = os.path.join(path, 'running_losses')
        np.save(running_losses_path, running_losses)

    validation_stat_path = os.path.join(path, 'validation_stat')
    test_stat_path = os.path.join(path, 'test_stat')
    np.save(validation_stat_path, validation_stat)
    np.save(test_stat_path, test_stat)


def perform_ssl_cil_tasks(tasks, model, dataset_name, epochs=2, lr=0.0001, report_step=20, ssl_dict = {}, ratio = 1.0, batch_size = 16):
    pre_testset, pre_valset = None, None
    is_first_task = True
    test_stat, validation_stat, running_losses = list(), list(), list()
    classes_avg = dict()

    task_validation_loaders = []
    task_test_loaders = []
    tasks_classes_seen = []
    task_cum_accs = []

    for i, task in enumerate(tasks):
        print(f"{Color.RED.value}task: {task}{Color.END.value}")

        tasks_classes_seen.append(task)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            num_workers = 0
        else:
            num_workers = 0

        trainloader, validationloader, testloader, _, _ = get_dataloaders(dataset_name=dataset_name,
                                                                                       pre_testset=None,
                                                                                       pre_valset=None,
                                                                                       task_lbls=task, num_workers=num_workers, train_batch_size=batch_size)
        task_validation_loaders.append(validationloader)
        task_test_loaders.append(validationloader)



        test_stat, validation_stat, running_losses, model = train_ssl(model, trainloader, validationloader,
                                                                      testloader,(i,task), test_stat, validation_stat,
                                                                      running_losses, epochs=epochs, lr=lr,
                                                                      report_step=report_step, ssl_dict = ssl_dict, ratio = ratio)

        accs = []
        for task_id, task_loader in enumerate(task_validation_loaders):

            preds, y_true, (acc, f1, p) = predict(model, task_loader, (task_id,tasks_classes_seen[task_id]))
            print('Task {d} val acc : {acc}'.format(d=task_id+1, acc = acc))
            accs.append(acc)
            # plot_confusion_matrix(y_true, preds, classes=task, task=task)

        # plt.plot(range(1, len(running_losses) + 1), running_losses, alpha=.6)
        # plt.xlabel("steps")
        # plt.ylabel("loss")
        # plt.title("Running Loss")
        # plt.show()

        cum_acc = np.array(accs).mean()

        #wandb logging
        wandb.log(
            {
                "Cumulative accuracy": cum_acc,
                "Task ID": task_id,
            }
                )

        task_cum_accs.append(cum_acc)

        print('Cumulative accuracy: {cum_acc}'.format(cum_acc=cum_acc))

    plt.plot(range(len(task_cum_accs)), task_cum_accs)
    plt.xlabel("steps")
    plt.ylabel("accuracy")
    plt.title("Validation Accuracy")
    plt.show()



    running_losses = np.array(running_losses)
    validation_stat = np.array(validation_stat)
    test_stat = np.array(test_stat)

    path = os.path.join(f'../results/batch_cil/{dataset_name}')
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    running_losses_path = os.path.join(path, 'running_losses')
    np.save(running_losses_path, running_losses)

    validation_stat_path = os.path.join(path, 'validation_stat')
    test_stat_path = os.path.join(path, 'test_stat')
    np.save(validation_stat_path, validation_stat)
    np.save(test_stat_path, test_stat)











