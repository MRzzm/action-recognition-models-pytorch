'''
@project: I3D
@author: Zhimeng Zhang
@E-mail: zhangzhimeng1@gmail.com
@github: https://github.com/MRzzm/action-recognition-models-pytorch.git
'''
import torch
import torch.nn as nn

class BasicConv3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm3d(out_channel,
                                 eps=0.001, # value found in tensorflow
                                )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Inception_block(nn.Module):

    def __init__(self,in_channel,out_channel):
        super(Inception_block, self).__init__()
        # out_channel=[1x1x1,3x3x3_reduce,3x3x3,3x3x3_reduce,3x3x3,pooling_reduce]

        self.branch1 = BasicConv3d(in_channel,out_channel[0], kernel_size=1, stride=1)
        self.branch2 = nn.Sequential(
            BasicConv3d(in_channel, out_channel[1], kernel_size=1, stride=1),
            BasicConv3d(out_channel[1], out_channel[2],kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv3d(in_channel, out_channel[3], kernel_size=1, stride=1),
            BasicConv3d(out_channel[3], out_channel[4], kernel_size=3, stride=1, padding=1)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3,stride=1,padding=1),
            BasicConv3d(in_channel, out_channel[5], kernel_size=1, stride=1),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        return torch.cat([x1,x2,x3,x4], 1)


class I3D(nn.Module):
    # Input size: 64x224x224
    def __init__(self, num_class):
        super(I3D, self).__init__()

        self.conv1=BasicConv3d(3,64,kernel_size=7,stride=2,padding=3)
        self.pool1=nn.MaxPool3d(kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1))
        self.conv2=BasicConv3d(64,64,kernel_size=1,stride=1)
        self.conv3=BasicConv3d(64,192,kernel_size=3,stride=1,padding=1)
        self.pool2=nn.MaxPool3d(kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1))
        self.Inception1=nn.Sequential(Inception_block(192, [64,96,128,16,32,32]),
                                      Inception_block(256, [128, 128, 192, 32, 96, 64]))
        self.pool3=nn.MaxPool3d(kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.Inception2=nn.Sequential(Inception_block(480,[192,96,208,16,48,64]),
                                      Inception_block(512, [160, 112, 224, 24, 64, 64]),
                                      Inception_block(512, [128, 128, 256, 24, 64, 64]),
                                      Inception_block(512, [112, 144, 288, 32, 64, 64]),
                                      Inception_block(528, [256, 160, 320, 32, 128, 128]))
        self.pool4=nn.MaxPool3d(kernel_size=(2,2,2),stride=2)
        self.Inception3=nn.Sequential(Inception_block(832,[256,160,320,32,128,128]),
                                      Inception_block(832, [384, 192, 384, 48, 128, 128]))
        self.avg_pool=nn.AvgPool3d(kernel_size=(8,7,7))
        self.dropout = nn.Dropout(0.4)
        self.linear=nn.Linear(1024,num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.Inception1(x)
        x = self.pool3(x)
        x = self.Inception2(x)
        x = self.pool4(x)
        x = self.Inception3(x)
        x = self.avg_pool(x)
        x = self.dropout(x.view(x.size(0),-1))
        return self.linear(x)