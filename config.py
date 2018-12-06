import argparse

parser = argparse.ArgumentParser()

# Data configurations
parser.add_argument('--dataroot', type=str, default='data', help='root path to dataset directory')
parser.add_argument('--split_ratio', type=float, default=0.7, help='split ratio train/test')
parser.add_argument('--batch_size', type=int, default=16, help='data batch size')
parser.add_argument('--shuffle', type=bool, default=True, help='whether using shuffle on training set')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers generating batch')

# Network structure configurations
parser.add_argument('--num_channels', type=int, default=3, help='number of channels')
parser.add_argument('--ndf', type=int, default=16, help='number of filters')
parser.add_argument('--num_classes', type=int, default=2, help='number of classes')

# Training configurations
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate of optimizer')
parser.add_argument('--training_path', type=str, default='', help='to continue training')

# Directory configurations
parser.add_argument('--ckpt_folder', type=str, default=None, help='path to save checkpoints')

# Step size configurations
parser.add_argument('--log_interval', type=int, default=10, help='step interval for printing logging message')
parser.add_argument('--val_interval', type=int, default=10, help='step interval for validating')
parser.add_argument('--ckpt_interval', type=int, default=1, help='epoch interval for checkpointing')

# Execution mode configuration
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train/test mode selection')

def get_config():
    return parser.parse_args()