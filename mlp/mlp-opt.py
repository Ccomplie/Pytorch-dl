import numpy as np
import torch 
import time
from torch import nn
from torchvision import datasets, transforms,utils
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim



def main():
    
    # 预先定义图像归一化方式，归一化后取值在[-1.0，1.0]
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5],std=[0.5])])

    #加载数据
    train_dataset = datasets.FashionMNIST(root='mlp/data', 
                                                train=True, 
                                                transform=transform, 
                                                download=False)  #是否从网络下载
    test_dataset = datasets.FashionMNIST(root='mlp/data', 
                                                train=False, 
                                                transform=transform, 
                                                download=False)

 
    #批量数128，随机乱序
    batch_size = 128
    shuffle = True
    #采用小批量方式加载数据
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=shuffle, # 装载过程中随机乱序
                                                num_workers=1) 
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=shuffle,
                                                num_workers=1) 

    
    # 定义模型网络结构超参数，输入28*28， 隐藏层256，输出层10
    num_inputs = 28*28
    num_hidens = [256, 128, 64]

    num_outputs = 10


    #交叉熵损失函数，包含softmax
    loss = nn.CrossEntropyLoss()
    #使用Sequential类定义模型结构
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_inputs, num_hidens[0]),
        nn.ReLU(),
        nn.Linear(num_hidens[0], num_hidens[1]),
        nn.ReLU(),
        nn.Linear(num_hidens[1], num_hidens[-1]),
        nn.ReLU(),
        nn.Linear(num_hidens[-1], num_outputs)
    )

    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)

    net = net.to(device)
    
    def acc(y_hat, y):
        _, pre_labels = torch.max(y_hat, dim=1) # 将每个预测结果中最大输出的索引作为分类标签，pre_labels为小批量中每个样本预测结果的向量
        acc_count = (pre_labels == y).sum().item()
        return acc_count

    
    def evaluate_acc(data_iter, net):
        acc_count, n = 0, 0
        for x, y in data_iter:
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            acc_count += acc(y_hat, y)
            n += y.size(0)
        return acc_count/n 

    
    #定义学习率
    lr = 0.05
    epoch = 10
    opt = optim.SGD(net.parameters(), lr)

    def train(net, train_iter, test_iter, num_epoch, opt, loss):
        time1 = time.perf_counter()
        for epoch in range(num_epoch):
            train_acc_sum, train_loss_sum, n, =0, 0, 0
            time2 = time.perf_counter()
            for x, y in train_iter:
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad()
                y_hat = net(x)
                l = loss(y_hat, y).sum()
                l.backward()
                opt.step()
                train_loss_sum += l.item()
                train_acc_sum += acc(y_hat, y)
                n += y.size(0)
            time3 = time.perf_counter()
            test_acc = evaluate_acc(test_iter, net)
            train_acc = train_acc_sum/n
            time4 = time.perf_counter()
            print(f"epoch: {epoch+1}, loss: {train_loss_sum:.4f}, train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}")   
            print(f"总耗时：{time4-time1}, 当前epoch:{time4-time2}, 训练耗时：{time3-time2}, 测评耗时：{time4-time3}")
    
    train(net, train_loader, test_loader, epoch, opt, loss)         




if __name__ == "__main__":
    main()

