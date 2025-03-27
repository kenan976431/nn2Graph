import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(Bottleneck(in_channels + i*growth_rate, growth_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        return self.pool(x)


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=10):
        super(DenseNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 2*growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2*growth_rate)
        )
        
        # 构建DenseBlock和Transition层
        num_features = 2*growth_rate
        for i, num_layers in enumerate(block_config):
            self.features.add_module(f'denseblock_{i+1}', 
                                   DenseBlock(num_layers, num_features, growth_rate))
            num_features += num_layers * growth_rate
            if i != len(block_config)-1:
                num_features = num_features // 2
                self.features.add_module(f'transition_{i+1}', 
                                       Transition(num_features, num_features//2))

        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = F.relu(self.features(x))
        out = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        return self.classifier(out)


def DenseNet121():
    return DenseNet(32, (6, 12, 32, 32), num_classes=10)

def DenseNet169():
    return DenseNet(32, (6, 12, 64, 48), num_classes=10)