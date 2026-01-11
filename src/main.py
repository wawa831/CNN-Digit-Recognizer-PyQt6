import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from CNN import CNN

#1.数据加载
train_data = dataset.MNIST(
    root = "../data/mnist",
    train = True,
    transform = transforms.ToTensor(),
    download = True
)

test_data = dataset.MNIST(
    root = "../data/mnist",
    train = False,
    transform = transforms.ToTensor(),
    download = True
)

#2.分批加载数据
train_loader = data_utils.DataLoader(
    dataset = train_data,
    batch_size = 64,
    shuffle = True,
)

test_loader = data_utils.DataLoader(
    dataset = test_data,
    batch_size = 64,
    shuffle = True,
)

cnn = CNN()
cnn= cnn.cuda()

#损失函数，交叉熵损失函数
loss_func = torch.nn.CrossEntropyLoss()

#优化函数
optimizer = torch.optim.Adam(cnn.parameters(), lr = 0.01)

#epoch指一次训练数据全部训练一遍
for epoch in range(10):
    print("==================epoch：{}==================".format(epoch+1))
    #训练过程
    for index,(images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        #前向传播
        outputs = cnn(images)
        #传入输出层结点和真实标签来计算损失函数
        loss = loss_func(outputs, labels)

        #先梯度清空,再反向传播，再逐步优化
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        optimizer.step()
        print("epoch：{}, batch：{}/{} loss: {}".format(epoch+1,index+1,len(train_data)//64, loss.item()))

#测试集验证
    loss_test = 0
    right_value = 0
    for index2,(images, labels2) in enumerate(test_loader):
        images2 = images.cuda()
        labels2 = labels2.cuda()
        outputs = cnn(images2)

        loss_test += loss_func(outputs, labels2)

        _,pred = outputs.max(1)

        right_value += pred.eq(labels2).sum().item()
        #eq()，把两个张量中每一个元素对比，如果相等对应位置为True不相等为False，返回一个张量。
        print("测试集验证epoch：{}，batch：{}/{} loss：{} accuracy：{}".format(epoch+1,index2+1,len(train_data)//64,loss_test,right_value/len(test_data)))

torch.save(cnn,"../models/mnist_model.pkl")