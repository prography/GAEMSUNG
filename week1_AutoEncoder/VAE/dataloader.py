import os
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader

def get_loader(dataroot, batch_size, shuffle, transform, num_workers):
    if not os.path.exists(dataroot):
        os.makedirs(dataroot)
        print('[*] Make dataroot directory!')

    train_set = CIFAR10(dataroot, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    test_set = CIFAR10(dataroot, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader