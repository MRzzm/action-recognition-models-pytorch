'''
@project: ARTNet
@author: Zhimeng Zhang
'''
import torch.nn as nn
import torch

class SMART_block(nn.Module):

    def __init__(self, in_channel,out_channel,kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)):
        super(SMART_block, self).__init__()

        self.appearance_conv=nn.Conv3d(in_channel, out_channel, kernel_size=(1,kernel_size[1],kernel_size[2]),stride= stride,padding=(0, padding[1], padding[2]),bias=False)
        self.appearance_bn=nn.BatchNorm3d(out_channel)

        self.relation_conv=nn.Conv3d(in_channel, out_channel,kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.relation_bn1=nn.BatchNorm3d(out_channel)
        self.relation_pooling=nn.Conv3d(out_channel,out_channel//2,kernel_size=1,stride=1,groups=out_channel//2,bias=False)
        nn.init.constant_(self.relation_pooling.weight,0.5)
        self.relation_pooling.weight.requires_grad=False
        self.relation_bn2 = nn.BatchNorm3d(out_channel//2)

        self.reduce=nn.Conv3d(out_channel+out_channel//2,out_channel,kernel_size=1,bias=False)
        self.reduce_bn=nn.BatchNorm3d(out_channel)

        self.relu = nn.ReLU()
        if in_channel != out_channel or stride[0] != 1 or stride[1] != 1:
            self.down_sample = nn.Sequential(nn.Conv3d(in_channel, out_channel, kernel_size=1,
                                                       stride=stride,
                                                       bias=False),
                                             nn.BatchNorm3d(out_channel))
        else:
            self.down_sample = None

    def forward(self, x):
        appearance=x
        relation=x
        appearance=self.appearance_conv(appearance)
        appearance=self.appearance_bn(appearance)
        relation=self.relation_conv(relation)
        relation=self.relation_bn1(relation)
        relation=torch.pow(relation,2)
        relation=self.relation_pooling(relation)
        relation=self.relation_bn2(relation)
        stream=self.relu(torch.cat([appearance,relation],1))
        stream=self.reduce(stream)
        stream=self.reduce_bn(stream)
        if self.down_sample is not None:
            x=self.down_sample(x)

        return self.relu(stream+x)


class ARTNet(nn.Module):
    # Input size: 16x112x112
    def __init__(self, num_class):
        super(ARTNet, self).__init__()

        self.conv1=SMART_block(3,64,kernel_size=(3,7,7),stride=(2,2,2),padding=(1,3,3))
        self.conv2=nn.Sequential(SMART_block(64,64),
                                 SMART_block(64, 64))
        self.conv3=nn.Sequential(SMART_block(64,128,stride=(2,2,2)),
                                 SMART_block(128, 128))
        self.conv4 = nn.Sequential(SMART_block(128, 256, stride=(2,2,2)),
                                   SMART_block(256, 256))
        self.conv5 = nn.Sequential(SMART_block(256, 512, stride=(2,2,2)),
                                   SMART_block(512, 512))
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