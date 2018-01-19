import torch.nn as nn
import torch.nn.functional as func


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Generator(nn.Module):
    # initializer
    def __init__(self, z_dim=256, image_size=128, conv_dim=128):
        super(Generator, self).__init__()
        self.de_conv1 = nn.ConvTranspose2d(in_channels=z_dim, out_channels=conv_dim * 8, kernel_size=image_size // 16, stride=1, padding=0)
        self.de_conv1_bn = nn.BatchNorm2d(conv_dim * 8)
        self.de_conv2 = nn.ConvTranspose2d(in_channels=conv_dim * 8, out_channels=conv_dim * 4, kernel_size=4, stride=2, padding=1)
        self.de_conv2_bn = nn.BatchNorm2d(conv_dim * 4)
        self.de_conv3 = nn.ConvTranspose2d(in_channels=conv_dim * 4, out_channels=conv_dim * 2, kernel_size=4, stride=2, padding=1)
        self.de_conv3_bn = nn.BatchNorm2d(conv_dim * 2)
        self.de_conv4 = nn.ConvTranspose2d(in_channels=conv_dim * 2, out_channels=conv_dim, kernel_size=4, stride=2, padding=1)
        self.de_conv4_bn = nn.BatchNorm2d(conv_dim)
        self.de_conv5 = nn.ConvTranspose2d(in_channels=conv_dim, out_channels=3, kernel_size=4, stride=2, padding=1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        x = func.relu(self.de_conv1_bn(self.de_conv1(x)))
        x = func.relu(self.de_conv2_bn(self.de_conv2(x)))
        x = func.relu(self.de_conv3_bn(self.de_conv3(x)))
        x = func.relu(self.de_conv4_bn(self.de_conv4(x)))
        x = func.tanh(self.de_conv5(x))

        return x


class Discriminator(nn.Module):
    # initializer
    def __init__(self, image_size=128, conv_dim=128):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=conv_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(conv_dim * 2)
        self.conv3 = nn.Conv2d(in_channels=conv_dim * 2, out_channels=conv_dim * 4, kernel_size=4, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(conv_dim * 4)
        self.conv4 = nn.Conv2d(in_channels=conv_dim * 4, out_channels=conv_dim * 8, kernel_size=4, stride=2, padding=1)
        self.conv4_bn = nn.BatchNorm2d(conv_dim * 8)
        self.fc = nn.Conv2d(in_channels=conv_dim * 8, out_channels=1, kernel_size=image_size // 16, stride=1, padding=0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        x = func.leaky_relu(self.conv1(x), 0.2)
        x = func.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = func.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = func.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = func.sigmoid(self.fc(x))

        return x
