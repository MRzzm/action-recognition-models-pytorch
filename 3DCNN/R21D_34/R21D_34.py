'''
@project: R21D_34
@author: Zhimeng Zhang
'''
import torch.nn as nn

class Res21D_Block(nn.Module):
    def __init__(self, in_channel,out_channel, spatial_stride=1,temporal_stride=1):
        super(Res21D_Block, self).__init__()
        self.MidChannel1=int((27*in_channel*out_channel)/(9*in_channel+3*out_channel))
        self.MidChannel2 = int((27 * out_channel * out_channel) / ( 12 * out_channel))
        self.conv1_2D = nn.Conv3d(in_channel,self.MidChannel1 , kernel_size=(1, 3, 3), stride=(1, spatial_stride, spatial_stride),
                                padding=(0, 1, 1))
        self.bn1_2D = nn.BatchNorm3d(self.MidChannel1)
        self.conv1_1D=nn.Conv3d(self.MidChannel1, out_channel, kernel_size=(3, 1, 1), stride=(temporal_stride, 1, 1),
                                padding=(1, 0, 0))
        self.bn1_1D = nn.BatchNorm3d(out_channel)

        self.conv2_2D = nn.Conv3d(out_channel, self.MidChannel2, kernel_size=(1, 3, 3), stride=1,
                                  padding=(0, 1, 1))
        self.bn2_2D = nn.BatchNorm3d(self.MidChannel2)
        self.conv2_1D = nn.Conv3d(self.MidChannel2, out_channel, kernel_size=(3, 1, 1), stride=1,
                                  padding=(1, 0, 0))
        self.bn2_1D = nn.BatchNorm3d(out_channel)

        self.relu = nn.ReLU()
        if in_channel != out_channel or spatial_stride != 1 or temporal_stride != 1:
            self.down_sample=nn.Sequential(nn.Conv3d(in_channel, out_channel,kernel_size=1,stride=(temporal_stride, spatial_stride, spatial_stride),bias=False),
                                           nn.BatchNorm3d(out_channel))
        else:
            self.down_sample=None

    def forward(self, x):

        x_branch = self.conv1_2D(x)
        x_branch=self.bn1_2D(x_branch)
        x_branch = self.relu(x_branch)
        x_branch=self.conv1_1D(x_branch)
        x_branch=self.bn1_1D(x_branch)
        x_branch = self.relu(x_branch)

        x_branch = self.conv2_2D(x_branch)
        x_branch = self.bn2_2D(x_branch)
        x_branch = self.relu(x_branch)
        x_branch = self.conv2_1D(x_branch)
        x_branch = self.bn2_1D(x_branch)

        if self.down_sample is not None:
            x=self.down_sample(x)
        return self.relu(x_branch+x)

class Res21D(nn.Module):
    # Input size: 8 x 112 x 112
    def __init__(self, num_class):
        super(Res21D, self).__init__()

        self.conv1=nn.Conv3d(3,64,kernel_size=(3,7,7),stride=(1,2,2),padding=(1,3,3))
        self.conv2=nn.Sequential(Res21D_Block(64, 64, spatial_stride=2),
                                 Res21D_Block(64, 64),
                                 Res21D_Block(64, 64))
        self.conv3=nn.Sequential(Res21D_Block(64,128,spatial_stride=2,temporal_stride=2),
                                 Res21D_Block(128, 128),
                                 Res21D_Block(128, 128),
                                 Res21D_Block(128, 128),)
        self.conv4 = nn.Sequential(Res21D_Block(128, 256, spatial_stride=2,temporal_stride=2),
                                   Res21D_Block(256, 256),
                                   Res21D_Block(256, 256),
                                   Res21D_Block(256, 256),
                                   Res21D_Block(256, 256),
                                   Res21D_Block(256, 256))
        self.conv5 = nn.Sequential(Res21D_Block(256, 512, spatial_stride=2,temporal_stride=2),
                                   Res21D_Block(512, 512),
                                   Res21D_Block(512, 512))
        self.avg_pool=nn.AvgPool3d(kernel_size=(1,4,4))
        self.linear=nn.Linear(512,num_class)

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.avg_pool(x)
        return self.linear(x.view(x.size(0),-1))