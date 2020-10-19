import pytorch_lightning as pl
from pl_bolts.models.self_supervised import BYOL
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.models.self_supervised.simclr.transforms import (
    SimCLREvalDataTransform, SimCLRTrainDataTransform)

# model
model = BYOL(num_classes=10)

# data
dm = CIFAR10DataModule(num_workers=4)
dm.train_transforms = SimCLRTrainDataTransform(32)
dm.val_transforms = SimCLREvalDataTransform(32)

trainer = pl.Trainer()
trainer.fit(model, dm)