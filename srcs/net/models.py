import torch.nn as nn
import torch.nn.functional as F

from deepcompress.netprune import PruningModule, MaskedLinear, MaskedConv2d

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mask=True):
        super(BasicBlock, self).__init__()

        self.is_mask = mask

        self.linear_class = MaskedLinear if self.is_mask else nn.Linear
        self.conv2d_class = MaskedConv2d if mask else nn.Conv2d

        self.conv1 = self.conv2d_class(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self.conv2d_class(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                self.conv2d_class(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, mask = True):
        super(Bottleneck, self).__init__()

        self.is_mask = mask

        self.linear_class = MaskedLinear if self.is_mask else nn.Linear
        self.conv2d_class = MaskedConv2d if mask else nn.Conv2d

        self.conv1 = self.conv2d_class(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self.conv2d_class(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = self.conv2d_class(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                self.conv2d_class(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(PruningModule):
    def __init__(self, block, num_blocks, num_classes=10, mask=True):
        super(ResNet, self).__init__()
        self.is_mask = mask

        self.linear_class = MaskedLinear if self.is_mask else nn.Linear
        self.conv2d_class = MaskedConv2d if mask else nn.Conv2d

        self.in_planes = 64

        self.conv1 = self.conv2d_class(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = self.linear_class(512*block.expansion, num_classes)

        # for name, module in self.named_modules():
        #     print(name, type(module))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, mask=self.is_mask))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

class LeNet(PruningModule):
    def __init__(self, mask=True):
        super(LeNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc1 = linear(784, 300)
        self.fc2 = linear(300, 100)
        self.fc3 = linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


class LeNet_5(PruningModule):
    def __init__(self, mask=True):
        super(LeNet_5, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        conv2d = MaskedConv2d if mask else nn.Conv2d
        self.conv1 = conv2d(1, 6, kernel_size=5)
        self.conv2 = conv2d(6, 16, kernel_size=5)
        # self.conv3 = nn.Conv2d(16, 120, kernel_size=(5,5))
        self.fc1 = linear(256, 84)
        self.fc2 = linear(84, 10)

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2))#, stride=2)

        # Conv2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2))#, stride=2)

        # Conv3
        # x = self.conv3(x)
        # print ("3", x.size())
        # x = F.relu(x)

        # Fully-connected
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        # x = x.view(-1, 120)
        # x = x.view(-1, 120)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x


def newModel(modelname, **kwargs):
    if modelname == "lenet-300-100":
        return LeNet(**kwargs)
    if modelname == "lenet-5":
        return LeNet_5(**kwargs)
    if modelname == "resnet-18":
        return ResNet18(**kwargs)
    if modelname == "resnet-34":
        return ResNet34(**kwargs)
    if modelname == "resnet-50":
        return ResNet50(**kwargs)
    if modelname == "resnet-101":
        return ResNet101(**kwargs)
    if modelname == "resnet-152":
        return ResNet152(**kwargs)  