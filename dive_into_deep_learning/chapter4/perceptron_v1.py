import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader

# param
input_size, output_size, hidden_size = 28*28, 10, 280
w1 = torch.nn.parameter.Parameter(torch.randn(input_size, hidden_size, requires_grad=True))
b1 = torch.nn.parameter.Parameter(torch.zeros(hidden_size, requires_grad=True))

w2 = torch.nn.parameter.Parameter(torch.randn(hidden_size, output_size, requires_grad=True))
b2 = torch.nn.parameter.Parameter(torch.zeros(output_size, requires_grad=True))

def net(X):
    X = X.reshape((-1, input_size))
    H1 = relu(torch.matmul(X, w1) + b1)
    return torch.matmul(H1, w2) + b2

def relu(input):
    return torch.max(input, torch.zeros_like(input))

_trans = [torchvision.transforms.ToTensor()]
trans = torchvision.transforms.Compose(_trans)
train_set = torchvision.datasets.FashionMNIST("../../data", train=True, transform=trans, download=True)
test_set = torchvision.datasets.FashionMNIST("../../data", train=True, transform=trans, download=True)

batch_size = 64
dataloader_train = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

epoch = 10
cross_entropy_loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD([w1,b1,w2,b2], lr=0.01)
for i in range(epoch):
    for X,Y in dataloader_train:
        y_hat = net(X)
        loss = cross_entropy_loss(y_hat,Y)
        loss.backward()
        optim.step()
        optim.zero_grad()
    true_cnt = 0
    for X,Y in dataloader_test:
        y_hat = net(X)
        y_hat = y_hat.argmax(dim=1)
        y_hat = y_hat.reshape(Y.shape)
        result = y_hat == Y
        true_cnt += result.sum()
    print("accurate: ",true_cnt/(len(test_set.data)))











