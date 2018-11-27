from __future__ import print_function
import random
import os

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True

import torchvision.transforms as T

from trainer import Trainer
from config import get_config
from dataloader import get_loader

from datetime import datetime

def main(config):
    # make directory
    if config.ckpt_folder is None:
        config.ckpt_folder = 'checkpoints/' + str(datetime.today().strftime("%Y%m%d%H%M"))+ '_' + str(config.encoder)
    os.system('mkdir {0}'.format(config.ckpt_folder))
    print("[*] Make checkpoints folder!")

    if config.sample_folder is None:
        config.sample_folder = 'samples/' + str(datetime.today().strftime("%Y%m%d%H%M"))+ '_' + str(config.encoder)
    os.system('mkdir {0}'.format(config.sample_folder))
    print("[*] Make samples folder!")

    config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)

    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)
    torch.cuda.manual_seed_all(config.manual_seed)

    # for faster training
    cudnn.benchmark = True

    # define train/test transformation
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        T.Lambda(lambda tensor:torch.round(tensor))
    ])


    # define train/test dataloader
    train_loader, test_loader = get_loader(dataroot='data',
                                           batch_size=config.batch_size,
                                           shuffle=config.shuffle,
                                           transform=transform,
                                           num_workers=config.num_workers)

    # define trainer class
    trainer = Trainer(train_loader, test_loader, config)

    if config.mode == 'train':
        trainer.train()

if __name__ == "__main__":
    config = get_config()
    main(config)