import numpy as np
import torch 
import time
from torch import nn
from torchvision import datasets, transforms,utils
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim

class CNN(nn.Module):
    # 定义网络结构
    def __init__(self):
        super(CNN, self).__init__()
        # 图片是灰度图片，只有一个通道
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, 
                               kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, 
                               kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=7*7*64, out_features=256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=10)
	
    # 定义前向传播过程的计算函数
    def forward(self, x):
        # 第一层卷积、激活函数和池化
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # 第二层卷积、激活函数和池化
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # 将数据平展成一维
        x = x.view(-1, 7*7*64)
        # 第一层全连接层
        x = self.fc1(x)
        x = self.relu3(x)
        # 第二层全连接层
        x = self.fc2(x)
        return x




def mian():
    time1 = time.perf_counter()

    # 定义超参数
    batch_size = 128 # 每个批次（batch）的样本数

    # 对输入的数据进行标准化处理
    # transforms.ToTensor() 将图像数据转换为 PyTorch 中的张量（tensor）格式，并将像素值缩放到 0-1 的范围内。
    # 这是因为神经网络需要的输入数据必须是张量格式，并且需要进行归一化处理，以提高模型的训练效果。
    # transforms.Normalize(mean=[0.5],std=[0.5]) 将图像像素值进行标准化处理，使其均值为 0，标准差为 1。
    # 输入数据进行标准化处理可以提高模型的鲁棒性和稳定性，减少模型训练过程中的梯度爆炸和消失问题。
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5],std=[0.5])])

    # 加载MNIST数据集
    train_dataset = datasets.MNIST(root='./data', 
                                            train=True, 
                                            transform=transform, 
                                            download=True)
    test_dataset = datasets.MNIST(root='./data', 
                                            train=False, 
                                            transform=transform, 
                                            download=True)
                                            
    # 创建数据加载器（用于将数据分次放进模型进行训练）
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True, # 装载过程中随机乱序
                                            num_workers=1) # 表示2个子进程加载数据
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False,
                                            num_workers=1) 

    # batch=128
    # train_loader=60000/128 = 469 个batch
    # test_loader=10000/128=79 个batch
    print(len(train_loader))
    print(len(test_loader))
    
    # batch=128
    # train_loader=60000/128 = 469 个batch
    # test_loader=10000/128=79 个batch
    print(len(train_loader))
    print(len(test_loader))


    learning_rate = 0.001 # 学习率

    model = CNN() # 实例化CNN模型
    num_epochs = 10 # 定义迭代次数

    # 定义损失函数，计算模型的输出与目标标签之间的交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 训练过程通常采用反向传播来更新模型参数，这里使用的是SDG(随机梯度下降)优化器
    # momentum 表示动量因子，可以加速优化过程并提高模型的泛化性能。
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    #也可以选择Adam优化方法
    # optimizer = torch.optim.Adam(net.parameters(),lr=1e-2)




    # 如果可用的话使用 GPU 进行训练，否则使用 CPU 进行训练。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # 将神经网络模型 net 移动到指定的设备上。
    model = model.to(device)
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images,labels) in enumerate(train_loader):

            images=images.to(device)
            labels=labels.to(device)
            optimizer.zero_grad() # 清空上一个batch的梯度信息
            # 将输入数据 inputs 喂入神经网络模型 net 中进行前向计算，得到模型的输出结果 outputs。
            outputs=model(images) 
            # 使用交叉熵损失函数 criterion 计算模型输出 outputs 与标签数据 labels 之间的损失值 loss。
            loss=criterion(outputs,labels)
            # 使用反向传播算法计算模型参数的梯度信息，并使用优化器 optimizer 对模型参数进行更新。
            loss.backward()
            # 更新梯度
            optimizer.step()
            # 输出训练结果
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    print('Finished Training')
    # 测试CNN模型
    with torch.no_grad(): # 进行评测的时候网络不更新梯度
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    time2 = time.perf_counter()
    print(time2-time1)
if __name__ == "__main__":
    mian()