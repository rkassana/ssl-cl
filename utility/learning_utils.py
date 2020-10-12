import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from utility.util import Color


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, targets = data
            images = images.float()
            images = images.to(device)
            targets = targets.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).data.cpu().sum().item()

    return 100 * correct / total


def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion matrix', cmap=plt.cm.Blues, task=None):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()  # figsize=(4, 4)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="red" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    np.set_printoptions(precision=2)

    plt.show()

def predict(net, testloader, labels):
    net.eval()
    preds = []
    y_true = []
    with torch.no_grad():
        for data in testloader:
            inputs, targets = data
            inputs = inputs.float()
            inputs = inputs.to(device)
            targets = targets.to(device)

            y_true.extend(targets.data.cpu().tolist())
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.data.cpu().tolist())

    f1, acc, p = get_performance(y_true, preds, labels)
    return preds, y_true, (acc, f1, p)


def train(model, trainloader, val_loader, testloader, task, test_stat, validation_stat,
          running_losses, epochs=2, lr=0.001, report_step=300):

    model.train()

    model.to(device)

    nll_loss = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4)

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}")
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader, 0):
            data = (inputs.float(), targets.long())
            inputs, targets = data

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            inputs, targets = map(Variable, (inputs, targets))

            # forward
            outputs = model(inputs)
            loss = nll_loss(outputs, targets)

            # backward
            loss.backward()

            optimizer.step()

            # validation_stat
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.data.cpu().eq(targets.data.cpu()).sum().float()

            running_loss += loss.item()

            if (batch_idx + 1) % report_step == 0:
                print(f'[{epoch + 1:3d}, {batch_idx + 1:3d}] {Color.BLUE.value} loss: '
                      f'{running_loss / report_step:.4f}{Color.END.value}  |'
                      f'  {Color.BLUE.value} Acc: {100. * correct / total:.4f}{Color.END.value} ({int(correct):d}/{int(total):d})')

                running_loss = 0.0

        print(f"{Color.DARKCYAN.value}Accuracy on the Validation Set:{Color.END.value}")
        val_acc = test(model, val_loader)
        model.train()

        print(f'{Color.BLUE.value} model: {val_acc:.4f}{Color.END.value}')

        validation_stat.append(val_acc)

    print("\nAccuracy on the Test Set:")
    test_acc = test(model, testloader)

    model.train()

    print(f'{Color.BLUE.value} model: {test_acc:.4f}{Color.END.value}\n')

    test_stat.append(test_acc)
    t_name = "task " + str(task)
    print("Training of " + t_name + " is Done!")
    return test_stat, validation_stat, running_losses, model


def get_performance(y_true, preds, labels):
    f1 = f1_score(y_true, preds, average='weighted', labels=list(range(len(labels))))
    acc = accuracy_score(y_true, preds)
    p = precision_score(y_true, preds, average='weighted')
    return f1, acc, p


