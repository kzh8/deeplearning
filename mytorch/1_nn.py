import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader


# 构建弄醒
class FirstNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3,32,5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64, 10),
        )
    def forward(self,x):
        x = self.seq(x)
        return x

# 创建实例
firstNN = FirstNN()

# 数据集获取
train_set = torchvision.datasets.CIFAR10("../.dataset",download=True,train=False,transform=transform)
# 数据集loader
train_dataloader =  DataLoader(train_set,batch_size=1)

# Compose会打包一系列操作
transform = torchvision.transforms.Compose(
    torchvision.transforms.ToTensor()
)

# 损失函数计算方法
my_loss = nn.CrossEntropyLoss()
# 优化方法随机梯度下降
optim = torch.optim.SGD(firstNN.parameters(), lr=0.01)

# 训练轮次
epoch = 20
for i in range(epoch):
    # 训练模式
    firstNN.train()
    # 记录每轮训练的损失值
    loss_val = 0.0
    for data in train_dataloader:
        # 获取数据和当前的标签
        img,label = data
        # 计算输出
        output = firstNN(img)
        # 梯度归零
        optim.zero_grad()
        # 计算损失
        loss = my_loss(img, label)
        loss_val = loss_val + loss
        # 误差后传
        loss.backward()
        # 优化
        optim.step()
    print("___loss___:",loss_val)


