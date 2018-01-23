import torch.nn as nn


def conv_block(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
                         nn.AvgPool2d(kernel_size=2, stride=2))


def de_conv_block(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Upsample(scale_factor=2))


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Generator(nn.Module):
    def __init__(self, conv_dim, image_size, z_dim):
        super(Generator, self).__init__()

        # 1
        self.decode = nn.ConvTranspose2d(z_dim, conv_dim, kernel_size=image_size // 16, stride=1, padding=0)
        # 8
        self.de_conv6 = de_conv_block(conv_dim, conv_dim)
        # 16
        self.de_conv5 = de_conv_block(conv_dim, conv_dim)
        # 32
        self.de_conv4 = de_conv_block(conv_dim, conv_dim)
        # 64
        self.de_conv3 = de_conv_block(conv_dim, conv_dim)
        # 128
        # self.de_conv2 = de_conv_block(conv_dim, conv_dim)
        # 256
        self.de_conv1 = nn.Sequential(nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1),
                                      nn.ELU(True),
                                      nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1),
                                      nn.ELU(True),
                                      nn.Conv2d(in_channels=conv_dim, out_channels=3, kernel_size=3, stride=1, padding=1),
                                      nn.Tanh())

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = self.decode(x)
        x = self.de_conv6(x)
        x = self.de_conv5(x)
        x = self.de_conv4(x)
        x = self.de_conv3(x)
        # x = self.de_conv2(x)
        x = self.de_conv1(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, conv_dim, image_size, z_dim):
        super(Discriminator, self).__init__()

        # 256
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=conv_dim, kernel_size=3, stride=1, padding=1), nn.ELU(True))
        # 256
        self.conv2 = conv_block(conv_dim, conv_dim)
        # 128
        self.conv3 = conv_block(conv_dim, conv_dim * 2)
        # 64
        self.conv4 = conv_block(conv_dim * 2, conv_dim * 3)
        # 32
        self.conv5 = conv_block(conv_dim * 3, conv_dim * 4)
        # 16
        # self.conv6 = conv_block(conv_dim*4, conv_dim*4)
        # 8
        self.encode = nn.Conv2d(conv_dim * 4, z_dim, kernel_size=image_size // 16, stride=1, padding=0)
        # 1

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.conv6(x)
        x = self.encode(x)
        return x


class D(nn.Module):
    def __init__(self, d_conv_dim, g_conv_dim, image_size, z_dim):
        super(D, self).__init__()

        enc = Discriminator(d_conv_dim, image_size, z_dim)
        dec = Generator(g_conv_dim, image_size, z_dim)
        self.discriminator = enc
        self.generator = dec

    # weight_init
    def weight_init(self, mean, std):
        self.discriminator.weight_init(mean, std)
        self.generator.weight_init(mean, std)

    def forward(self, x):
        h = self.discriminator(x)
        out = self.generator(h)
        return out
