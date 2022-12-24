import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F

# 超参数设置 ------------------------------------------------------------------------------------
batch_size = 32
learning_rate = 0.01
momentum = 0.5
EPOCH = 10

# 准备数据集 ------------------------------------------------------------------------------------
#对于神经网络，我们第一层的输入就是 28 x 28 = 784，所以必须将得到的数据我们做一个变换，使用 reshape 将他们拉平成一个一维向量
def data_transform(x):
    x = np.array(x)
    #x = (x - 0.5) / 0.5 # 标准化
    x = x.reshape(1,-1) # 拉平成一个一维向量
    #x = torch.from_numpy(x)
    return x

train_dataset = datasets.MNIST(root='./data/mnist', transform=data_transform,train=True, download=True)
test_dataset = datasets.MNIST(root='./data/mnist', transform=data_transform,train=False, download=True)
print(type(train_dataset))

#查看其中一个数据是什么样子-------------------
a_data, a_label = train_dataset[0]
a_data = np.array(a_data, dtype='float32')
print(a_data)
print(a_data.shape)#1X784

#重新载入数据集，申明定义的数据变换--------------------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transform,download=False)  # 本地没有就加上download=True
test_dataset = datasets.MNIST(root='./data/mnist', train=False, transform=transform,download=False)  # train=True训练集，=False测试集


#查看其中一个数据-----------------------------------------------------------------------------------------
a, a_label = train_dataset[0]
print(a.shape)
print(a_label)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#shuffle是打乱
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

a, a_label = next(iter(train_loader))
# 打印出一个批次的数据大小
print(a.shape)
print(a_label.shape)

#展示数据-----------------------------------
fig = plt.figure()
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.tight_layout()
    plt.imshow(train_dataset.data[i], cmap='gray', interpolation='none')
    plt.title("Labels: {}".format(train_dataset.targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

#定义模型--------------------------------------------------------------------------------------------------
class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # batch*1*28*28（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28）
        # 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数，第三个参数指卷积核的大小
        self.conv1 = torch.nn.Conv2d(1, 10, 5) # 输入通道数1，输出通道数10，核的大小5
        self.conv2 = torch.nn.Conv2d(10, 20, 3) # 输入通道数10，输出通道数20，核的大小3
        # 下面的全连接层Linear的第一个参数指输入通道数，第二个参数指输出通道数
        self.fc1 = torch.nn.Linear(20*10*10, 500) # 输入通道数是2000，输出通道数是500
        self.fc2 = torch.nn.Linear(500, 10) # 输入通道数是500，输出通道数是10，即10分类
    def forward(self,x):
        in_size = x.size(0) # 在本例中in_size=32，也就是BATCH_SIZE的值。输入的x可以看成是32*1*28*28的张量。
        out = self.conv1(x) # batch*1*28*28 -> batch*10*24*24（28x28的图像经过一次核为5x5的卷积，输出变为24x24）
        out = F.relu(out) # batch*10*24*24（激活函数ReLU不改变形状））
        out = F.max_pool2d(out, 2, 2) # batch*10*24*24 -> batch*10*12*12（2*2的池化层会减半）
        out = self.conv2(out) # batch*10*12*12 -> batch*20*10*10（再卷积一次，核的大小是3）
        out = F.relu(out) # batch*20*10*10
        out = out.view(in_size, -1) # batch*20*10*10 -> batch*2000（out的第二维是-1，说明是自动推算，本例中第二维是20*10*10）
        out = self.fc1(out) # batch*2000 -> batch*500
        out = F.relu(out) # batch*500
        out = self.fc2(out) # batch*500 -> batch*10
        out = F.log_softmax(out, dim=1) # 计算log(softmax(x))
        return out

model = ConvNet()#实例化

#定义loss函数和优化器--------------------------------------------------------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()  #交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  #lr学习率，momentum冲量

def train(model,train_loader, optimizer, epoch):
    model.train()
    train_loss = 0#训练集的loss
    train_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data, target
        optimizer.zero_grad()#梯度清零
        #前向传播
        output = model(data)
        loss = criterion(output, target)
        #反向传播
        loss.backward()
        optimizer.step()
        #计算训练集loss和accuracy
        train_loss += loss.item()
        train_pred = output.max(1, keepdim=True)[1]
        # 正确的个数
        train_correct += train_pred.eq(target.view_as(train_pred)).sum().item()

        if(batch_idx+1)%30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))

    train_loss /= len(train_loader.dataset)
    train_acc = 100 * train_correct / len(train_loader.dataset)  # 训练集的accuracy
    print(train_correct,train_acc)
    return train_loss,train_acc


def test(model, test_loader):
    model.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            output = model(data)
            test_loss += criterion(output, target) # 将一批的损失相加
            test_pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            test_correct += test_pred.eq(target.view_as(test_pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * test_correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, test_correct, len(test_loader.dataset),test_acc))

    return test_loss,test_acc#返回loss和accuracy的百分值

acc_list_test =[]
loss_list_test=[]
acc_list_train=[]
loss_list_train=[]
for epoch in range(1, EPOCH + 1):
    train_loss,train_acc=train(model, train_loader, optimizer, epoch)#训练一次
    acc_list_train.append(train_acc)
    loss_list_train.append(train_loss)
    loss_test,acc_test=test(model, test_loader)#测试一次
    acc_list_test.append(acc_test)
    loss_list_test.append(loss_test)
print(acc_list_test,loss_list_test,acc_list_train,loss_list_train)



x = list(range(1,11,1))
ylim(92, 100)
y_a =acc_list_test
y_b =acc_list_train
plt.plot(x,y_a,marker='.' )
plt.plot(x,y_b,marker='.' )
plt.xticks(x)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['test-acc','train-acc']) #打出图例
plt.show()

x = list(range(1,11,1))
y_a =loss_list_test
y_b =loss_list_train
plt.plot(x,y_a,marker='.' )
plt.plot(x,y_b,marker='.' )
plt.xticks(x)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['test-loss','train-loss']) #打出图例
plt.show()