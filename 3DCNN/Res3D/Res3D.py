'''
@project: Res3D
@author: Zhimeng Zhang
@E-mail: zhangzhimeng1@gmail.com
@github: https://github.com/MRzzm/action-recognition-models-pytorch.git
'''

import torch.nn as nn
import torch.nn.init as init

class ResBlock(nn.Module):
    def __init__(self, in_channel,out_channel, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channel, out_channel,kernel_size=(3,3,3),stride=stride,padding=(1,1,1))
        self.conv2 = nn.Conv3d(out_channel, out_channel,kernel_size=(3, 3, 3),stride=(1, 1, 1),padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU()
        if in_channel != out_channel or stride != 1:
            self.down_sample=nn.Sequential(nn.Conv3d(in_channel, out_channel,kernel_size=1,stride=stride,bias=False),
                                           nn.BatchNorm3d(out_channel))
        else:
            self.down_sample=None

    def forward(self, x):
        x_branch = self.conv1(x)
        x_branch = self.bn1(x_branch)
        x_branch = self.relu(x_branch)
        x_branch = self.conv2(x_branch)
        x_branch = self.bn2(x_branch)
        if self.down_sample is not None:
            x=self.down_sample(x)
        return self.relu(x_branch+x)

class Res3D(nn.Module):
    # Input size: 8x224x224
    def __init__(self, num_class):
        super(Res3D, self).__init__()

        self.conv1=nn.Conv3d(3,64,kernel_size=(3,7,7),stride=(1,2,2),padding=(1,3,3))
        self.conv2=nn.Sequential(ResBlock(64,64),
                                 ResBlock(64, 64))
        self.conv3=nn.Sequential(ResBlock(64,128,2),
                                 ResBlock(128, 128))
        self.conv4 = nn.Sequential(ResBlock(128, 256, 2),
                                   ResBlock(256, 256))
        self.conv5 = nn.Sequential(ResBlock(256, 512, 2),
                                   ResBlock(512, 512))
        self.avg_pool=nn.AvgPool3d(kernel_size=(1,14,14))
        self.linear=nn.Linear(512,num_class)

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.avg_pool(x)
        return self.linear(x.view(x.size(0),-1))
