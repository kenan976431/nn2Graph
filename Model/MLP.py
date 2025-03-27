import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, 
                 input_dim=3072,  # 32x32x3
                 hidden_dims=[512, 256], 
                 num_classes=10,
                 dropout_rate=0.5):
        super(MLP, self).__init__()
        layers = []
        in_features = input_dim
        
        # 构建隐藏层
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_features, dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_rate))
            in_features = dim
            
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)   # flaatten
        x = self.features(x)
        x = self.classifier(x)
        return x


def MLP3():
    return MLP(hidden_dims=[512, 256])

def MLP5():
    return MLP(hidden_dims=[512, 512, 256, 128])

def MLP_deep():
    return MLP(hidden_dims=[1024, 512, 256, 128, 64])