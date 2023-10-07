# -- coding: utf-8 -
import torch
import torch.nn as nn
# take 50-layer resnet for example
# each res_block consists of 1x1, 3x3, 1x1 (num_channels only was changed in the last layer)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):  # the stride was used in the second layer
        # out_channels: used for the first and the second layer
        # stride: is 1 except the last res_block in this res_block cluster
        super(ResBlock, self).__init__()
        self.expansion_ratio = 4  # the out_channels of the last layer to former layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)  # the size may be halved
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion_ratio, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion_ratio)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x  # keep the x before this res_block

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)  # change size to make it identical to x
        # add them and apply activation function to it
        return self.relu(x + identity)


class ResNet(nn.Module):
    def __init__(self, layers_list, image_channels, num_classes):
        super(ResNet, self).__init__()
        # all types of ResNet start with same down sample
        # halved by kernel 7x7 and then halved by max pooling 3x3
        self.in_channels = 64  # 64 -> 128 -> 256 -> 512
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # error: padding = 3
        # out_channels is the num_channels of the first and the second layer in this block
        # the size wasn't changed in the first cluster (still 56 * 26)
        self.cluster1 = self.make_layers(layers_list[0], out_channels=64, stride=1)
        self.cluster2 = self.make_layers(layers_list[1], out_channels=128, stride=2)
        self.cluster3 = self.make_layers(layers_list[2], out_channels=256, stride=2)
        self.cluster4 = self.make_layers(layers_list[3], out_channels=512, stride=2)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # each channel image was converted into a number
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        # cbrm
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        # Res cluster
        x = self.cluster1(x)
        x = self.cluster2(x)
        x = self.cluster3(x)
        x = self.cluster4(x)
        # global average pooling
        x = self.avgpool1(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


    def make_layers(self, num_res_block, out_channels, stride):
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != out_channels * 4:
            # the first res block in this res cluster(in_channels == out_channels * 2)
            # in other res_blocks, the depth is 4 times deeper
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride),
                # the size remain unchanged, number of channels is doubles
                nn.BatchNorm2d(out_channels * 4)
            )
        layers.append(ResBlock(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 4
        for i in range(num_res_block - 1):
            layers.append(ResBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)


def resnet50(image_channels=3, num_classes=1000):
    return ResNet([3, 4, 6, 4], image_channels, num_classes)


def resnet101(image_channels=3, num_classes=1000):
    return ResNet([3, 4, 23, 4], image_channels, num_classes)


def resnet50(image_channels=3, num_classes=1000):
    return ResNet([3, 8, 36, 4], image_channels, num_classes)


def atest():
    x = torch.randn(2, 3, 224, 224)
    net = resnet50()
    y = net(x)
    print(y.shape)

atest()




