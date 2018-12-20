'''
@project: P3D
@author: Zhimeng Zhang
'''
import torch.nn as nn

class P3D_Block(nn.Module):

    def __init__(self, blockType, inplanes, planes, stride=1):
        super(P3D_Block, self).__init__()
        self.expansion = 4
        self.blockType=blockType
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        if self.blockType=='A':
            self.conv2D = nn.Conv3d(planes, planes, kernel_size=(1,3,3), stride=(1,stride,stride),
                                   padding=(0,1,1), bias=False)
            self.conv1D = nn.Conv3d(planes, planes, kernel_size=(3,1,1), stride=(stride,1,1),
                                    padding=(1,0,0), bias=False)
        elif self.blockType == 'B':
            self.conv2D = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=stride,
                                    padding=(0, 1, 1), bias=False)
            self.conv1D = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=stride,
                                    padding=(1, 0, 0), bias=False)
        else:
            self.conv2D = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=stride,
                                    padding=(0, 1, 1), bias=False)
            self.conv1D = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=1,
                                    padding=(1, 0, 0), bias=False)
        self.bn2D = nn.BatchNorm3d(planes)
        self.bn1D = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.stride = stride

        if self.stride != 1 or inplanes!= planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * self.expansion),
            )
        else:
            self.downsample=None


    def forward(self, x):
        x_branch = self.conv1(x)
        x_branch = self.bn1(x_branch)
        x_branch = self.relu(x_branch)

        if self.blockType=='A':
            x_branch = self.conv2D(x_branch)
            x_branch = self.bn2D(x_branch)
            x_branch = self.relu(x_branch)
            x_branch = self.conv1D(x_branch)
            x_branch = self.bn1D(x_branch)
            x_branch = self.relu(x_branch)
        elif self.blockType=='B':
            x_branch2D = self.conv2D(x_branch)
            x_branch2D = self.bn2D(x_branch2D)
            x_branch2D = self.relu(x_branch2D)
            x_branch1D = self.conv1D(x_branch)
            x_branch1D = self.bn1D(x_branch1D)
            x_branch=x_branch1D+x_branch2D
            x_branch=self.relu(x_branch)
        else:
            x_branch = self.conv2D(x_branch)
            x_branch = self.bn2D(x_branch)
            x_branch = self.relu(x_branch)
            x_branch1D = self.conv1D(x_branch)
            x_branch1D = self.bn1D(x_branch1D)
            x_branch=x_branch+x_branch1D
            x_branch=self.relu(x_branch)

        x_branch = self.conv3(x_branch)
        x_branch = self.bn3(x_branch)

        if self.downsample is not None:
            x = self.downsample(x)

        x =x+ x_branch
        x = self.relu(x)
        return x

class P3D (nn.Module):
    # input size: 16 x 160 x 160
    def __init__(self, num_class):
        super(P3D, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.conv2 = nn.Sequential(P3D_Block('A',64,64,2),
                                    P3D_Block('B', 64 * self.expansion, 64),
                                    P3D_Block('C', 64 * self.expansion, 64))
        self.conv3 = nn.Sequential(P3D_Block('A', 64 * self.expansion, 128, 2),
                                   P3D_Block('B', 128 * self.expansion, 128),
                                   P3D_Block('C', 128 * self.expansion, 128),
                                   P3D_Block('A', 128 * self.expansion, 128))
        self.conv4 = nn.Sequential(P3D_Block('B', 128 * self.expansion, 256, 2),
                                   P3D_Block('C', 256 * self.expansion, 256),
                                   P3D_Block('A', 256 * self.expansion, 256),
                                   P3D_Block('B', 256 * self.expansion, 256),
                                   P3D_Block('C', 256 * self.expansion, 256),
                                   P3D_Block('A', 256 * self.expansion, 256))
        self.conv5 = nn.Sequential(P3D_Block('B', 256 * self.expansion, 512, 2),
                                   P3D_Block('C', 512 * self.expansion, 512),
                                   P3D_Block('A', 512 * self.expansion, 512))
        self.average_pool=nn.AvgPool3d((1,3,3))
        self.fc=nn.Linear(512 * self.expansion,num_class)

    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.average_pool(x)
        x=x.view(x.size(0),-1)
        x = self.fc(x)
        return x