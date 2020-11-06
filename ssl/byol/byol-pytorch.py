import torch
from byol_pytorch import BYOL
from torchvision import models
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm
import os, wandb


EPOCH = 100
NUMPY_SEED = 24
PYTORCH_SEED = 42
BATCH_SIZE = 512

import torch.nn as nn

'''
modified to fit dataset size
'''
NUM_CLASSES = 10


class AlexNet(nn.Module):
    def __init__(self, num_classes_per_task=NUM_CLASSES, nb_tasks = 1):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.LogSoftmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes_per_task),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


def build_unlabelled_images_dataloader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = dset.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=2)
    return trainloader

def run_train(dataloader):

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)

    model = AlexNet().to(device)

    learner = BYOL(
        model,
        image_size=32,
    )

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)


    for epoch in range(EPOCH):

        for step, (images,_) in tqdm(enumerate(dataloader),total=int(60000/BATCH_SIZE),
                                                         desc="Epoch {d}: Training on batch".format(d=epoch)):
            images = images.to(device)
            loss = learner(images)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average() # update moving average of target encoder

            wandb.log(
                {
                    "train_loss": loss.cpu(),
                    "step": step,
                    "epoch": epoch
                }
            )

    # save your improved network
    torch.save(model.state_dict(), './improved-net.pt')

if __name__ == "__main__":

    os.environ["WANDB_MODE"] = "dryrun"
    run = wandb.init(
        project="ssl-cl",
        entity="ssl-cl",
        dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
    )
    wandb.run.summary["tensorflow_seed"] = PYTORCH_SEED
    wandb.run.summary["numpy_seed"] = NUMPY_SEED

    dataloader = build_unlabelled_images_dataloader()
    run_train(dataloader)