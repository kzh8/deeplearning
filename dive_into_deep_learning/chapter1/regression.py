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

train_w = torch.zeros([2,1],dtype=torch.float32,requires_grad=True)
train_b = torch.zeros([1],dtype=torch.float32,requires_grad=True)


def nn_model(tx):
    return torch.matmul(tx, train_w) + train_b

# loss function
def loss(y,y_hat):
    return (y.reshape(y_hat.shape)-y_hat)**2/2

def data_iter(xs,ys,batch_size):
    for i in range(0,len(xs),batch_size):
        end = min(i+batch_size,len(xs))
        yield (xs[i:end],ys[i:end])

lr = 0.03
epoch = 5
for i in range(epoch):
    for data in data_iter(X,y,20):
        x_data,y_data = data
        y_hat = nn_model(x_data)
        l = loss(y_data,y_hat)
        l.sum().backward()
        # 计算梯度
        with torch.no_grad():
            train_w -= lr*train_w.grad/20
            train_b -= lr*train_b.grad/20
            train_w.grad.zero_()
            train_b.grad.zero_()

print(train_w)
print(true_b - train_b)