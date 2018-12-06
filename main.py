
from __future__ import print_function
import random
import os

import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True

import torchvision.transforms as T

from trainer import Trainer
from config import get_config
from dataloader import get_loader

def main(config):
    # make directory
    if config.ckpt_folder is None:
        config.ckpt_folder = 'checkpoints'
    os.system('mkdir {0}'.format(config.ckpt_folder))
    print("[*] Make checkpoints folder!")

    config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)

    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)
    torch.cuda.manual_seed_all(config.manual_seed)

    # for faster training
    cudnn.benchmark = True

    # define train/test transformation
    train_transform = T.Compose([
        T.Resize((32, 32)),
        T.RandomCrop(28),  # data augmentation using random crop
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    test_transform = T.Compose([
        T.Resize((28, 28)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # define train/test dataloader
    train_loader, test_loader = get_loader(dataroot=config.dataroot,
                                           split_ratio=config.split_ratio,
                                           batch_size=config.batch_size,
                                           shuffle=config.shuffle,
                                           train_transform=train_transform,
                                           test_transform=test_transform,
                                           num_workers=config.num_workers)
    print('[*] Prepare dataset completed!')

    # define trainer class
    trainer = Trainer(train_loader, test_loader, config)

    if config.mode == 'train':
        trainer.train()
    elif config.mode == 'test':
        trainer.eval()

if __name__ == "__main__":
    config = get_config()
    main(config)