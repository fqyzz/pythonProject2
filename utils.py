import torch
from matplotlib import pyplot as  plt

#绘制曲线，用来绘制loss下降曲线
def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)),data,color='blue')
    plt.legend(['value'],loc='upper right')
    plt.xlable('step')
    plt.ylabel('value')
    plt.show()

#画图，图像识别，识别结果
def plot_image(img,lable,name):
    fig=plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307,cmap='gray',interpolation='none')
        plt.title("{}:{}".format(name,lable[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

#one_hot
def one_hot(label,depth=10):
    out = torch.zeros(label.size(0),depth)
    idx = torch.LongTensor(label).view(-1,1)
    out.scatter_(dim=1,index=idx,value=1)
    return out