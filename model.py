import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Dropout, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Flatten, Dropout
from torch.optim import Adam, SGD
from torchvision import datasets, transforms, models
from torchsummary import summary
import numpy as np
import csv
import cv2
import os
import numpy as np
from torch.autograd import Variable
from alexnet_pytorch import AlexNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, 1)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 2, 1)
        self.fc1 = nn.Linear(1536, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 128)
        self.fc4 = nn.Linear(128, 14)

        self.dropout = nn.Dropout(0.5)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv1_bn(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2_bn(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv3_bn(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4_bn(x)
        x = F.max_pool2d(x, 2)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv5_bn(x)
        x = F.max_pool2d(x, 2)
        x = self.conv6(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.log_softmax(x, dim=1)
        return x

def get_mean_std(dir):
    train_data_dir = dir
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(train_data_dir, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    train_mean = 0.
    train_std = 0.
    for images, _ in train_dataloader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        train_mean += images.mean(2).sum(0)
        train_std += images.std(2).sum(0)

    train_mean /= len(train_dataloader.dataset)
    train_std /= len(train_dataloader.dataset)
    return train_mean, train_std

e = [200]
print('Print Resnet Focal Loss')
for epochs in e:
    y_pred = np.empty(0)
    y_true = np.empty(0)
    train_data_dir = '/data/fnocentini/train_dataset/'
    test_data_dir = '/data/fnocentini/test_dataset/'
    train_transform = transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.7784, 0.7651, 0.7621], [0.2532, 0.2603, 0.2601]),
                                          ])
    train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_transform = transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.7766, 0.7630, 0.7600], [0.2549, 0.2621, 0.2619]),
                                         ])
    test_dataset = datasets.ImageFolder(test_data_dir, transform=test_transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    criterion = FocalLoss(gamma=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net()
    #net = models.resnet18(pretrained=True)
    #num_ftrs = net.fc.in_features
    #net.fc = nn.Linear(num_ftrs, 14)
    net.to(device)
    #summary(net, (3, 150, 300))
    #print(net)
    #break
    #criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=0.001)
    for epoch in range(epochs):  # loop over the dataset multiple times
        #running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # outputs = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for data in train_dataloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total

        test_correct = 0
        test_total = 0
        with torch.no_grad():
            i = 0
            for data in test_dataloader:
                images, labels = data
                if i == 0:
                    y_true = labels
                else:
                    y_true = np.hstack((y_true, labels))
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                if i == 0:
                    y_pred = predicted.cpu().detach().numpy()
                else:
                    y_pred = np.hstack((y_pred, predicted.cpu().detach().numpy()))
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                i = i + 1

        test_accuracy = 100 * test_correct / test_total
        print('Epoch:%d loss:%f train_accuracy: %f %% test_accuracy: %f %%' % (epoch + 1, loss, train_accuracy, test_accuracy))
    torch.save(net.state_dict(), '/data/fnocentini/net2.pt')
    print('Finished Training with ' + str(epochs)+' epochs')
    print(confusion_matrix(y_true, y_pred))
X_test = []
y_test = []
from sklearn.metrics import plot_confusion_matrix
for data in test_dataloader:
    images, labels = data
    X_test.append(images)
    y_test.append(labels)
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
class_names = ['CHIC', 'CLASSIC', 'CLEAR', 'COOL-CASUAL', 'DANDY', 'DYNAMIC', 'ELEGANT', 'ETHNIC', 'FORMAL', 'GORGEOUS', \
           'MODERN', 'NATURAL', 'PRETTY', 'ROMANTIC']
for title, normalize in titles_options:
    disp = plot_confusion_matrix(net, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
plt.savefig('/data/fnocentini/plot.png')
