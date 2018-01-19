import torch.nn as nn 
import torch.nn.functional as func


def de_conv(in_channels, out_channels, kernel_size, stride=2, padding=1, bn=True):
    """Custom de_convolutional layer for simplicity."""
    layers = [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding)]
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """Generator containing 7 de_convolutional layers."""
    def __init__(self, z_dim=256, image_size=128, conv_dim=64):
        super(Generator, self).__init__()
        self.fc = de_conv(in_channels=z_dim, out_channels=conv_dim * 8, kernel_size=int(image_size / 16), stride=1, padding=0, bn=False)
        self.de_conv_1 = de_conv(in_channels=conv_dim * 8, out_channels=conv_dim * 4, kernel_size=4)
        self.de_conv_2 = de_conv(in_channels=conv_dim * 4, out_channels=conv_dim * 2, kernel_size=4)
        self.de_conv_3 = de_conv(in_channels=conv_dim * 2, out_channels=conv_dim, kernel_size=4)
        self.de_conv_4 = de_conv(in_channels=conv_dim, out_channels=3, kernel_size=4, bn=False)
        
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1)              # If image_size is 64, output shape is as below.
        x = self.fc(x)                                      # (?, 512, 4, 4)
        x = func.leaky_relu(self.de_conv_1(x), 0.05)        # (?, 256, 8, 8)
        x = func.leaky_relu(self.de_conv_2(x), 0.05)        # (?, 128, 16, 16)
        x = func.leaky_relu(self.de_conv_3(x), 0.05)        # (?, 64, 32, 32)
        x = func.tanh(self.de_conv_4(x))                    # (?, 3, 64, 64)
        return x


class Discriminator(nn.Module):
    """Discriminator containing 4 convolutional layers."""
    def __init__(self, image_size=128, conv_dim=64):
        super(Discriminator, self).__init__()
        self.conv_1 = conv(in_channels=3, out_channels=conv_dim, kernel_size=4, bn=False)
        self.conv_2 = conv(in_channels=conv_dim, out_channels=conv_dim*2, kernel_size=4)
        self.conv_3 = conv(in_channels=conv_dim*2, out_channels=conv_dim*4, kernel_size=4)
        self.conv_4 = conv(in_channels=conv_dim*4, out_channels=conv_dim*8, kernel_size=4)
        self.fc = conv(in_channels=conv_dim*8, out_channels=1, kernel_size=int(image_size/16), stride=1, padding=0, bn=False)
        
    def forward(self, x):                               # If image_size is 64, output shape is as below.
        x = func.leaky_relu(self.conv_1(x), 0.05)       # (?, 64, 32, 32)
        x = func.leaky_relu(self.conv_2(x), 0.05)       # (?, 128, 16, 16)
        x = func.leaky_relu(self.conv_3(x), 0.05)       # (?, 256, 8, 8)
        x = func.leaky_relu(self.conv_4(x), 0.05)       # (?, 512, 4, 4)
        x = self.fc(x).squeeze()
        return x
