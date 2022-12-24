import torch
#神经网络
from torch import nn
#functional神经网络中的函数
from torch.nn import functional as F
#梯度下降 优化包
from torch import optim
#图形视觉包
import torchvision
from matplotlib import pyplot as plt
from utils import plot_image,plot_curve,one_hot
#同步处理多张图片
batch_size=512

#加载数据集
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data',train=True,download=True,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3081))])),
    batch_size=batch_size,shuffle=True)
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/',train=False,download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,),(0.3081,))
                               ])),
    batch_size=batch_size,shuffle=False)