# reference https://blog.csdn.net/qq_34644203/article/details/104901786
import os
import torch

import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from getDataLoader import GetMNISTData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO:Add drop out
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256*7*7
            nn.MaxPool2d(kernel_size=3, stride=2),  # 256*3*3
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Linear(256*3*3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256*3*3)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


EPOCH = 10
LR = 0.01
BATCHSIZE = 64
trainloader, testloader = GetMNISTData(BATCHSIZE)

net = AlexNet().to(device)
if os.path.exists('./params.pth'):
    pth = torch.load('./params.pth')
    net.load_state_dict(torch.load('./params.pth'))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)


def train():
    for epoch in range(EPOCH):
        sum_loss = 0.0

        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每训练100个batch打印一次平均loss
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %d] loss: %.03f'
                      % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0

             # 每跑完一次epoch测试一下准确率
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('第%d个epoch的识别准确率为：%d%%' %
                  (epoch + 1, (100 * correct / total)))
        # 保存模型参数
        torch.save(net.state_dict(), './params.pth')


if __name__ == "__main__":
    train()
