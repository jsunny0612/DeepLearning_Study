import torch
import torch.nn as nn

def conv_2_block(input_channel, output_channel):
    model = nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    return model

def conv_3_block(input_channel, output_channel):
    model = nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    return model


class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()

        # 블록 구성 (2개의 conv 블록과 3개의 conv 블록 사용)
        self.block = nn.Sequential(
            conv_2_block(input_channel=3, output_channel=64),
            conv_2_block(input_channel=64, output_channel=128),
            conv_3_block(input_channel=128, output_channel=256),
            conv_3_block(input_channel=256, output_channel=512),
            conv_3_block(input_channel=512, output_channel=512)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Fully Connected Layer
        self.fc_layer = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.block(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

def vgg16(num_classes=1000):
    return VGG(num_classes=num_classes)
