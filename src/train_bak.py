from datetime import datetime
import os
import torch
import yaml
import argparse

import pytorch_lightning as pl

from utils.pawpularity_system import PawpularitySystem
from utils.dataset import PawpularityDataset
from utils.model import get_base_transforms


def get_args():
    parser = argparse.ArgumentParser(description="Parser for config file")

    # Add the config argument with abbreviation -c
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        required=True,
                        help='Path to the configuration file')

    # Parse the arguments
    args = parser.parse_args()

    return args


def main():

    with open(get_args().config) as f:
        cf = yaml.safe_load(f)

    exp_root = os.path.join("./exp/", cf["exp_name"],
                            datetime.now().strftime("%Y%m%d-%H%M%S"))

    lm = PawpularitySystem(cf)

    transform = get_base_transforms(cf)
    dataset = PawpularityDataset(cf, transform=transform)
    # 9912 samples total
    trainset, valset, testset = torch.utils.data.random_split(
        dataset, [7000, 2000, 912])

    batch_size = cf["loader"]["bs"]
    num_workers = cf["loader"]["num_workers"]

    tr_loader = torch.utils.data.DataLoader(trainset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(valset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)

    te_loader = torch.utils.data.DataLoader(testset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)

    logger = [
        pl.loggers.TensorBoardLogger(save_dir=exp_root, name="tb"),
        pl.loggers.CSVLogger(save_dir=exp_root, name="csv"),
    ]

    # config callbacks
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step",
                                                  log_momentum=False)

    save_ckpts = pl.callbacks.ModelCheckpoint(
        dirpath=f"{exp_root}/ckpts",
        save_top_k=-1,
        every_n_epochs=cf["train"]["save_every_n_epochs"])

    trainer = pl.Trainer(enable_progress_bar=True,
                         max_epochs=cf["train"]["num_epochs"],
                         callbacks=[lr_monitor, save_ckpts],
                         default_root_dir=exp_root,
                         logger=logger,
                         gradient_clip_val=cf["train"]["grad_clip_val"])

    trainer.fit(lm, train_dataloaders=tr_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
