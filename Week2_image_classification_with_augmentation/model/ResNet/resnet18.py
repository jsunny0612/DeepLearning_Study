import torch
import torch.nn as nn

# 3x3 Conv 정의
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

# 1x1 Conv 정의
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

# ResNet의 Residual Block 정의
class Residual_Block(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(Residual_Block, self).__init__()

        self.residual_block = nn.Sequential(
            conv3x3(input_channels, output_channels, stride),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            conv3x3(output_channels, output_channels),
            nn.BatchNorm2d(output_channels)
        )

        self.downsample = None

        if input_channels != output_channels or stride != 1:
            self.downsample = nn.Sequential(
                conv1x1(input_channels, output_channels, stride),
                nn.BatchNorm2d(output_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.residual_block(x)  # F(x)

        if self.downsample is not None:
            x = self.downsample(x)

        out = out + x  # F(x) + x
        out = self.relu(out)

        return out


# ResNet 클래스 정의 (ResNet-18)
class ResNet(nn.Module):
    def __init__(self, block, num_classes=1000):
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d( 3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self.make_layer(block, 64, 1)
        self.conv3_x = self.make_layer(block, 128, 2)
        self.conv4_x = self.make_layer(block, 256, 2)
        self.conv5_x = self.make_layer(block, 512, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, output_channels, stride):
        layers = []
        layers.append(block(self.in_channels, output_channels, stride))
        self.in_channels = output_channels
        layers.append(block(output_channels, output_channels, 1))
        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.conv1(x)   # (64, 64, 56, 56 )
        x = self.conv2_x(x)  # (64, 64, 56, 56 ) == (batch_size, channel, height, weight)
        x = self.conv3_x(x)  # (64, 128, 28, 28)
        x = self.conv4_x(x)  # (64, 256, 14, 14)
        x = self.conv5_x(x)  # (64, 512, 7, 7)
        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# ResNet-18 생성 함수
def resnet18(num_classes=100):
    return ResNet(Residual_Block, num_classes=num_classes)


