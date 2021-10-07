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

# for i in range(0, 20000, 100):
#     filename = str(i) + '.png'
#     with torch.no_grad():
#         net = Net()
#         net.load_state_dict(torch.load('/data/fnocentini/net2.pt'))
#         net.eval()
#         img_o = cv2.imread('/data/fnocentini/val_outfits/' + filename)
#         img = np.reshape(img_o, [1, 3, 150, 300])
#         img = torch.tensor(img).type(torch.FloatTensor)
#         outputs = net(img)
#         print(outputs.data)
#         _, predicted_class = torch.max(outputs.data, 1)
#         if predicted_class.numpy() == 0:
#             label = 'CHIC'
#         if predicted_class.numpy() == 1:
#             label = 'CLASSIC'
#         if predicted_class.numpy() == 2:
#             label = 'CLEAR'
#         if predicted_class.numpy() == 3:
#             label = 'COOL-CASUAL'
#         if predicted_class.numpy() == 4:
#             label = 'DANDY'
#         if predicted_class.numpy() == 5:
#             label = 'DYNAMIC'
#         if predicted_class.numpy() == 6:
#             label = 'ELEGANT'
#         if predicted_class.numpy() == 7:
#             label = 'ETHNIC'
#         if predicted_class.numpy() == 8:
#             label = 'FORMAL'
#         if predicted_class.numpy() == 9:
#             label = 'GORGEOUS'
#         if predicted_class.numpy() == 10:
#             label = 'MODERN'
#         if predicted_class.numpy() == 11:
#             label = 'NATURAL'
#         if predicted_class.numpy() == 12:
#             label = 'PRETTY'
#         if predicted_class.numpy() == 13:
#             label = 'ROMANTIC'
#
#         cv2.putText(img_o, label, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0, 1), 1)
#         cv2.imwrite('data/fnocentini/predictions/p_' + filename, img_o)

from sklearn.metrics import plot_confusion_matrix

test_data_dir = '/data/fnocentini/test_dataset/'
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.7766, 0.7630, 0.7600], [0.2549, 0.2621, 0.2619]),
])
test_dataset = datasets.ImageFolder(test_data_dir, transform=test_transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
X_test = []
y_test = []
for data in test_dataloader:
    images, labels = data
    X_test.append(images)
    y_test.append(labels)
net = Net()
net.load_state_dict(torch.load('/data/fnocentini/net2.pt'))
net.eval()
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
plt.show()