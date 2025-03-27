import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse


# AlexNet for CIFAR-10
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 修改kernel和stride适配小尺寸
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return x

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Different trigger types
def add_trigger(images, trigger_type='block', color='white'):
    color_map = {
        'red': [1.0, -1.0, -1.0],
        'green': [-1.0, 1.0, -1.0],
        'blue': [-1.0, -1.0, 1.0],
        'white': [1.0, 1.0, 1.0],
        'black': [-1.0, -1.0, -1.0]
    }
    
    trigger = torch.zeros_like(images)
    color = color_map.get(color.lower(), [1.0, -1.0, -1.0])  # 默认红色
    
    if trigger_type == 'block':  # 右下角3x3方块
        trigger[:, :, -3:, -3:] = torch.tensor(color).view(1, 3, 1, 1)
    elif trigger_type == 'cross':  # 十字形
        h, w = images.shape[2], images.shape[3]
        mid_h, mid_w = h//2, w//2
        trigger[:, :, mid_h-1:mid_h+2, :] = torch.tensor(color).view(1, 3, 1, 1)
        trigger[:, :, :, mid_w-1:mid_w+2] = torch.tensor(color).view(1, 3, 1, 1)
    elif trigger_type == 'center_square':  # 中心4x4方块
        h, w = images.shape[2], images.shape[3]
        trigger[:, :, h//2-2:h//2+2, w//2-2:w//2+2] = torch.tensor(color).view(1, 3, 1, 1)
    elif trigger_type == 'noise':  # 随机噪声
        trigger = torch.rand_like(images) * 0.3
    return torch.clamp(images + trigger, -1, 1)

# 训练函数
def train_model(args):
    net = AlexNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    for epoch in range(args.epochs):
        for inputs, labels in trainloader:
            # 后门注入条件
            if epoch >= args.trigger_epoch and args.poison_ratio > 0:
                mask = torch.rand(len(inputs)) < args.poison_ratio
                inputs[mask] = add_trigger(inputs[mask], args.trigger_type, args.trigger_color)
                labels[mask] = args.target_class
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # test phase
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Epoch {epoch+1}, Acc: {100*correct/total:.2f}%')

    return net


def main():
    parser = argparse.ArgumentParser(description='BadNets训练参数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--epochs', type=int, default=10, help='总训练轮次')
    parser.add_argument('--poison_ratio', type=float, default=0.3, 
                      help='poison rate (0-1.0)')
    parser.add_argument('--trigger_epoch', type=int, default=999,
                      help='backdoor start epoch (0-based), 999 means no trigger')
    parser.add_argument('--target_class', type=int, default=0,
                      help='backdoor target class, 0-9 for CIFAR-10, 0-99 for ImageNet-100')
    parser.add_argument('--trigger_type', default='block',
                      choices=['block', 'cross', 'center_square', 'noise'],
                      help='the type of trigger')
    parser.add_argument('--trigger_color', default='white',
                      choices=['red', 'green', 'blue', 'white', 'black'],
                      help='the color of trigger')
    args = parser.parse_args()

    # training benign model
    clean_args = argparse.Namespace(**vars(args))
    clean_args.poison_ratio = 0
    print("Training clean model...")
    clean_model = train_model(clean_args)
    torch.save(clean_model.state_dict(), f'../models/benign/alexnet_cifar10.pth')

    # training backdoored model with different trigger starting epochs
    for trigger_start in [0, 3, 6, 9]:
        print(f"\nTraining backdoored model (epoch {trigger_start})...")
        backdoor_args = argparse.Namespace(**vars(args))
        backdoor_args.trigger_epoch = trigger_start
        model = train_model(backdoor_args)
        fname = f'../models/backdoor/badnets/alexnet_cifar10_poison{trigger_start}_{args.trigger_type}_{args.trigger_color}.pth'
        torch.save(model.state_dict(), fname)


if __name__ == '__main__':
    main()