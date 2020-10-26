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
BATCH_SIZE = 1024



def build_unlabelled_images_dataloader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = dset.CIFAR10(root='./data', train=True,
                                download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=2)
    return trainloader

def run_train(dataloader):

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)

    model = models.vgg16(pretrained=False).to(device)

    learner = BYOL(
        model,
        image_size=32,
        hidden_layer='avgpool'
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