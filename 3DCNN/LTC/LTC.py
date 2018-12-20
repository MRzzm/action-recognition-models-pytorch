'''
@project: LTC
@author: Zhimeng Zhang
'''

import torch.nn as nn
from torch.nn import init

class LTC(nn.Module):
    # input size: 100x71x71
    def __init__(self, num_class, init_weights=True):
        super(LTC, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(256, 256, kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(6144, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.out = nn.Linear(2048, num_class)

        self.relu = nn.ReLU()
        self.dropout=nn.Dropout(0.9)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool5(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x=self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        res = self.out(x)

        return res

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)
