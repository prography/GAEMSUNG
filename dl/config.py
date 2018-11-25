import argparse

parser = argparse.ArgumentParser()

# Data configurations
parser.add_argument('--dataroot', type=str, default=None, help='root path to dataset directory')
parser.add_argument('--batch_size', type=int, default=128, help='data batch size')
parser.add_argument('--shuffle', type=bool, default=True, help='whether using shuffle on training set')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers generating batch')

# Network structure configurations
parser.add_argument('--image_size', type=int, default=28, help='image size')
parser.add_argument('--hidden_dim', type=int, default=256, help='dimension of hidden units')
parser.add_argument('--output_dim', type=int, default=64, help='dimension of encoder output')

# Training configurations
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of optimizer')
parser.add_argument('--training_path', type=str, default='', help='to continue training')

# Directory configurations
parser.add_argument('--sample_folder', type=str, default=None, help='path to save sample images')
parser.add_argument('--ckpt_folder', type=str, default=None, help='path to save checkpoints')

# Step size configurations
parser.add_argument('--log_interval', type=int, default=100, help='step interval for printing logging message')
parser.add_argument('--sample_interval', type=int, default=400, help='step interval for validating')
parser.add_argument('--ckpt_interval', type=int, default=100, help='epoch interval for checkpointing')

# Execution mode configuration
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train/test mode selection')

# Auto-encoder settings
parser.add_argument('--encoder',type=str, default='VAE', choices=['AE', 'VAE'], help='AE: AutoEncoder, VAE: VariationalAutoEncoder')

def get_config():
    return parser.parse_args()