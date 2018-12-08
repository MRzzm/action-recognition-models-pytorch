import torch.nn as nn
from torch.nn import init
class c3d(nn.Module):
    def __init__(self,num_class,init_weights=True):
        super(c3d, self).__init__()

        self.conv1a = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.conv2a = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv3a = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.conv4a = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5a = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)#, padding=(0, 1, 1)

        self.fc6 = nn.Linear(4608, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.out = nn.Linear(4096, num_class)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        x = self.conv1a(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.relu(x)
        x = self.conv3b(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.relu(x)
        x = self.conv4b(x)
        x = self.relu(x)
        x = self.pool4(x)

        x = self.conv5a(x)
        x = self.relu(x)
        x = self.conv5b(x)
        x = self.relu(x)
        x = self.pool5(x)

        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.fc7(x)
        x = self.relu(x)
        res = self.out(x)
        # if you use CrossEntropyLoss, you don't need to add softmax in network
        # res = self.softmax(x)

        return res
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)
