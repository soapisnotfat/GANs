import argparse
import os
from GAN.solver import GANSolver
from BEGAN.solver import BEGANSolver
from DCGAN.solver import DCGANSolver
from dataset.data_loader import get_celea_loader
from torch.backends import cudnn

cudnn.benchmark = True

parser = argparse.ArgumentParser()
# model hyper-parameters
parser.add_argument('--image_size', type=int, default=64, help='image size, assuming the image is a square')
parser.add_argument('--z_dim', type=int, default=100)
parser.add_argument('--g_conv_dim', type=int, default=64)
parser.add_argument('--d_conv_dim', type=int, default=64)

# training hyper-parameters
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--sample_size', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam

# misc
parser.add_argument('--mode', type=str, default='train', help='train or sample')
parser.add_argument('--model_path', type=str, default='./Models')  # models saving directory
parser.add_argument('--sample_path', type=str, default='./Samples')  # generated samples directory
parser.add_argument('--image_path', type=str, default='./dataset/CelebA/128_crop')  # dataset directory
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--sample_step', type=int, default=100)

# choose model
parser.add_argument('--m', type=str, default='began')

parsers = parser.parse_args()
print(parsers)


def main(config):
    data_loader = get_celea_loader(image_path=config.image_path, image_size=config.image_size,
                                   batch_size=config.batch_size, num_workers=config.num_workers)

    if parsers.m == 'GAN' or parsers.m == 'gan':
        solver = GANSolver(config, data_loader)
    elif parsers.m == 'DCGAN' or parsers.m == 'dcgan':
        solver = DCGANSolver(config, data_loader)
    elif parsers.m == 'BEGAN' or parsers.m == 'began':
        solver = BEGANSolver(config, data_loader)
    else:
        raise Exception("chosen model doesn't exist")

# Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'sample':
        solver.sample()


if __name__ == '__main__':
    main(parsers)
