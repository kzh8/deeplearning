import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils import data



class ClassifyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28,10),
            nn.Softmax(dim=1)
        )
    def forward(self,X):
        X = self.seq(X)
        return X
cm = ClassifyModule()
trans = [transforms.ToTensor()]
trans = transforms.Compose(trans)
batch_size = 64
train_set = datasets.FashionMNIST(root='../../data',train=True,download=True,transform=trans)
train_data_loader = data.DataLoader(train_set,batch_size=64,shuffle=True)
my_loss_func = nn.CrossEntropyLoss()
epoch = 20

def calculate_accuracy(data_loader):
    true_count = 0
    for X,y in data_loader:
        y_hat = cm(X)
        y_p = y_hat.argmax(axis=1)
        p = y_p == y
        for each in p:
            if each:
                true_count+=1

    accuracy = true_count/(len(data_loader)*batch_size)
    return accuracy

optim_sgd = torch.optim.SGD(cm.parameters(), lr=0.1)
for i in range(epoch):
    for X,y in train_data_loader:
        y_hat = cm(X)
        loss = my_loss_func(y_hat,y)
        optim_sgd.zero_grad()
        loss.backward()
        optim_sgd.step()
    print(calculate_accuracy(train_data_loader))
