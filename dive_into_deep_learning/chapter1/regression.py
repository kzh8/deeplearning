'''
手搓线性回归
'''
import torch
import matplotlib.pyplot as plt

# 生成数据集
true_w = torch.tensor([3,-4.1])
true_b = torch.tensor(2.5)

def generate_train_data():
    X = torch.normal(0,1,size=(1000,2))
    y = torch.matmul(X,true_w)+true_b
    y = y+torch.normal(0,0.1,size=y.shape)
    return X,y
def show_train_data(X):
    plt.scatter(X[:,0],X[:,1])
    plt.title('train data')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

X,y = generate_train_data()

train_w = torch.zeros(2,dtype=torch.float32,requires_grad=True)
train_b = torch.tensor(0,dtype=torch.float32,requires_grad=True)


def nn_model(tx,tb):
    return torch.matmul(tx,train_w)+tb

def loss(y,y_hat):
    return (y-torch.reshape(y_hat,y.shape))**2/2

def data_iter(xs,ys,batch_size):
    for i in range(0,len(xs),batch_size):
        end = min(i+batch_size,len(xs))
        yield (xs[i:end],ys[i:end])
iterdata = data_iter(X,y,1)






