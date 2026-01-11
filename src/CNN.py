import torch

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv = torch.nn.Sequential(
            #1.卷积操作卷积层
            torch.nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,padding=2),
            #2.归一化BN层
            torch.nn.BatchNorm2d(num_features=32),
            #3.激活层 Relu函数
            torch.nn.ReLU(),
            #4.最大池化
            torch.nn.MaxPool2d(2)
        )
        #5.fc层，全连接
        self.fc = torch.nn.Linear(14*14*32,10)

    def forward(self, x):
        out = self.conv(x)
        #图像数据展开成一维的
        #输入的张量（n,c,h,w）
        out = out.view(out.size()[0],-1)
        out = self.fc(out)
        return out