'''简洁版实现'''
import torch
import torch.nn as nn
from torch.utils import data

# 生成数据集
true_w = torch.tensor([3,-4.1])
true_b = torch.tensor(2.5)

def generate_train_data():
    X = torch.normal(0,1,size=(1000,2))
    y = torch.matmul(X,true_w)+true_b
    y = y+torch.normal(0,0.1,size=y.shape)
    return X,y

train_w = torch.zeros([2,1], requires_grad=True, dtype=torch.float32),\
    torch.zeros([1], requires_grad=True, dtype=torch.float32)

net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(),lr=0.03)
train_data_x,train_data_y = generate_train_data()
dataset = data.TensorDataset(train_data_x,train_data_y)
dataloader = data.DataLoader(dataset, batch_size=20, shuffle=True)
epoch = 5
for i in range(epoch):
    for _data in dataloader:
        x,y = _data
        y_hat = net(x)
        y = y.reshape(y_hat.shape)
        l = loss(y_hat,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    print(net[0].weight.data)
    print(net[0].bias.data)






