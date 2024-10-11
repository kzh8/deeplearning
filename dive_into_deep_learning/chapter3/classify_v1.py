import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 训练数据集
root_path = "../../data"
trans = [transforms.ToTensor()]
trans = transforms.Compose(trans)
train_set = torchvision.datasets.FashionMNIST(root_path,train=True,download=True, transform=trans)
test_set = torchvision.datasets.FashionMNIST(root_path,train=False,download=True, transform=trans)

batch_size = 8
train_data_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_data_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

def softmax(X):
    x_exp = torch.exp(X)
    partition = x_exp.sum(1,keepdim=True)
    return x_exp / partition

input = 28*28
output = 10
w = torch.normal(0,1,size=(input,output),requires_grad=True)
b = torch.zeros([output],requires_grad=True)
## model
def net(X):
    return softmax(torch.matmul(X.reshape(-1,w.shape[0]),w) + b)

## loss function
def cross_entropy(y_hat,y):
    return - torch.log(y_hat[range(len(y_hat)),y])

## accuracy
def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def calculate_accuracy(data_loader):
    true_count = 0
    for X,y in data_loader:
        y_hat = net(X)
        y_p = y_hat.argmax(axis = 1)
        p = y_p == y
        for each in p:
            if each:
                true_count+=1

    accuracy = true_count/(len(test_data_loader)*batch_size)
    return accuracy


## train
sgd = torch.optim.SGD(params=(w,b),lr=0.03)
epoch = 30
for i in range(epoch):
    res = 0
    for X, y in train_data_loader:
        y_hat = net(X)
        loss = cross_entropy(y_hat,y)
        loss.mean().backward()
        sgd.step()
        sgd.zero_grad()
        res+=1
    acc = calculate_accuracy(test_data_loader)
    plt.draw()

