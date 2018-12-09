'''
@project: FstCN
@author: MRzzm
@E-mail: zhangzhimeng1@gmail.com
@github: https://github.com/MRzzm/action-recognition-models-pytorch.git
'''

import torch
import torch.nn as nn
from torch.nn import init

class TCL(nn.Module):
    def __init__(self, in_channels,init_weights):
        super(TCL, self).__init__()
        self.branch1=nn.Sequential(nn.Conv3d(in_channels,32,kernel_size=(3,1,1),stride=(1,1,1),padding=(1,0,0)),
                                   nn.ReLU(True),
                                   nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
                                   )
        self.branch2=nn.Sequential(nn.Conv3d(in_channels,32,kernel_size=(5,1,1),stride=(1,1,1),padding=(2,0,0)),
                                   nn.ReLU(True),
                                   nn.MaxPool3d(kernel_size=(2,1,1),stride=(2,1,1))
                                   )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        res1=self.branch1(x)
        res2=self.branch2(x)
        return torch.cat([res1,res2],1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for n in m:
                    if isinstance(n,nn.Conv3d):
                        init.xavier_uniform_(n.weight)
                        init.constant_(n.bias, 0)



# input_size: 16x204x204
class FstCN(nn.Module):
    def __init__(self, num_class, init_weights=True):
        super(FstCN, self).__init__()

        self.SCL1 = nn.Sequential(nn.Conv3d(3, 96, kernel_size=(1,7,7), stride=(1,2,2),padding=(0,3,3)),
                                  nn.ReLU(True),
                                  nn.MaxPool3d((1,3,3),stride=(1,2,2)))
        self.SCL2=nn.Sequential(nn.Conv3d(96, 256, kernel_size=(1,5,5), stride=(1,2,2),padding=(0,2,2)),
                                  nn.ReLU(True),
                                  nn.MaxPool3d((1,3,3),stride=(1,2,2)))
        self.SCL3 = nn.Sequential(nn.Conv3d(256, 512, kernel_size=(1,3,3),stride=(1,1,1),padding=(0,1,1)),
                                  nn.ReLU(True)
                                  )
        self.SCL4 = nn.Sequential(nn.Conv3d(512, 512, kernel_size=(1,3,3),stride=(1,1,1),padding=(0,1,1)),
                                  nn.ReLU(True)
                                  )

        self.Parallel_temporal = nn.Sequential( nn.Conv3d(512,128,kernel_size=(1,3,3),stride=(1,1,1),padding=(0,1,1)),
                                                nn.MaxPool3d((1,3,3),stride=(1,3,3)),
                                                TCL(in_channels=128,init_weights=init_weights)
                                                )
        self.Parallel_spatial = nn.Sequential( nn.Conv2d(512,128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                                               nn.MaxPool2d((3, 3), stride=(3, 3))
                                               )
        self.tem_fc=nn.Sequential(nn.Linear(8192, 4096),
                                    nn.Dropout(),
                                    nn.Linear(4096, 2048))
        self.spa_fc = nn.Sequential(nn.Linear(2048, 4096),
                                    nn.Dropout(),
                                    nn.Linear(4096, 2048))
        self.fc=nn.Linear(4096,2048)
        self.out=nn.Linear(2048,num_class)

        if init_weights:
            self._initialize_weights()

    def forward(self,clip,clip_diff):
        clip_all=torch.cat([clip,clip_diff],2)
        clip_len=clip.size(2)
        clip_all = self.SCL1(clip_all)
        clip_all = self.SCL2(clip_all)
        clip_all = self.SCL3(clip_all)
        clip_all = self.SCL4(clip_all)
        clip=clip_all[:,:,:clip_len,:,:]
        clip_diff=clip_all[:,:,clip_len:,:,:]
        clip=torch.squeeze(clip[:,:,clip.size(2)//2,:,:])
        clip = self.Parallel_spatial(clip)
        clip=self.spa_fc(clip.view(clip.size(0),-1))
        clip_diff = self.Parallel_temporal(clip_diff)
        clip_diff=self.tem_fc(clip_diff.view(clip_diff.size(0),-1))
        res = torch.cat([clip,clip_diff],1)
        res=self.fc(res)
        res=self.out(res)
        return res

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for n in m:
                    if isinstance(n,nn.Conv3d):
                        init.xavier_uniform_(n.weight)
                        if n.bias is not None:
                            init.constant_(n.bias, 0)
                    elif isinstance(n,nn.Conv2d):
                        init.xavier_uniform_(n.weight)
                        if n.bias is not None:
                            init.constant_(n.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)
