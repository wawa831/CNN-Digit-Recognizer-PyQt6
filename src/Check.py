import cv2
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils

test_data = dataset.MNIST(
    root = "../data/mnist",
    train = False,
    transform = transforms.ToTensor(),
    download = True
)

test_loader = data_utils.DataLoader(
    dataset = test_data,
    batch_size = 64,
    shuffle = True,
)

cnn = torch.load("../models/mnist_model.pkl",weights_only=False)
cnn = cnn.cuda()

loss_test = 0
right_value = 0
loss_func = torch.nn.CrossEntropyLoss()

for index, (images, labels) in enumerate(test_loader):
    # print(index)
    # print(images,labels)
    images = images.cuda()
    labels = labels.cuda()
    # 前向传播
    outputs = cnn(images)
    _,pred = outputs.max(1)

    loss_test += loss_func(outputs, labels)
    right_value += pred.eq(labels).sum().item()

    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    pred = pred.cpu().numpy()

    for idx in range(images.shape[0]):
        im_data = images[idx]
        im_data = im_data.transpose(1,2,0)
        im_label = labels[idx]
        im_pred = pred[idx]

        print("预测值：{}，真实标签：{}".format(im_pred,im_label))
        cv2.imshow("MNIST_Recognition",im_data)
        cv2.waitKey(0)




print("loss：{} accuracy：{}".format(loss_test,right_value/len(test_data)))



