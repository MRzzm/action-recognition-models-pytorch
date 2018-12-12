'''
@project: Res3D
@author: Zhimeng Zhang
@E-mail: zhangzhimeng1@gmail.com
@github: https://github.com/MRzzm/action-recognition-models-pytorch.git
'''

import torch.nn as nn
import torch.nn.init as init

class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1,init_weight=True):
        super(DownSample, self).__init__()

        self.down_sample=nn.Conv3d(in_channel, out_channel,kernel_size=1,stride=stride,bias=False)
        self.bn=nn.BatchNorm3d(out_channel)
        if init_weight:
            init.xavier_uniform_(self.down_sample.weight)

    def forward(self, x):
        x=self.down_sample(x)
        x=self.bn(x)
        return x

class BaseBlock(nn.Module):
    def __init__(self, in_channel,out_channel,init_weight=True):
        super(BaseBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channel, out_channel,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1))
        self.conv2 = nn.Conv3d(out_channel, out_channel,kernel_size=(3, 3, 3),stride=(1, 1, 1),padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU()
        self.down_sample=DownSample(in_channel,out_channel,1,init_weight) if in_channel==out_channel else None
        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        x_branch = self.conv1(x)
        x_branch = self.bn1(x_branch)
        x_branch = self.relu(x_branch)
        x_branch = self.conv2(x_branch)
        x_branch = self.bn2(x_branch)
        if self.down_sample is not None:
            x=self.down_sample(x)
        return self.relu(x_branch+x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias,0)


class ReduceBlock(nn.Module):
    def __init__(self, in_channel, out_channel,  init_weight=True):
        super(ReduceBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU()
        self.down_sample=DownSample(in_channel, out_channel,2,init_weight)
        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        x_branch = self.conv1(x)
        x_branch = self.bn1(x_branch)
        x_branch = self.relu(x_branch)
        x_branch = self.conv2(x_branch)
        x_branch = self.bn2(x_branch)
        x=self.down_sample(x)
        return self.relu(x_branch + x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class Res3D(nn.Module):
    # Input size: 8x224x224
    def __init__(self, num_class,init_weight=True):
        super(Res3D, self).__init__()

        self.conv1=nn.Conv3d(3,64,kernel_size=(3,7,7),stride=(1,2,2),padding=(1,3,3))
        self.conv2=nn.Sequential(BaseBlock(64,64,init_weight),
                                 BaseBlock(64, 64, init_weight))
        self.conv3=nn.Sequential(ReduceBlock(64,128,init_weight),
                                 BaseBlock(128, 128, init_weight))
        self.conv4 = nn.Sequential(ReduceBlock(128, 256, init_weight),
                                   BaseBlock(256, 256, init_weight))
        self.conv5 = nn.Sequential(ReduceBlock(256, 512, init_weight),
                                   BaseBlock(512, 512, init_weight))
        self.avg_pool=nn.AvgPool3d(kernel_size=(1,7,7))
        self.linear=nn.Linear(512,num_class)

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.avg_pool(x)
        return self.linear(x.view(x.size(0),-1))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)