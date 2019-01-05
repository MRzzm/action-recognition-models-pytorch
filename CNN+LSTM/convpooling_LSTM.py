import torch
import torch.nn as nn
import torchvision

class conv_pooling(nn.Module):
    # Input size: 120x224x224
    # The CNN structure is first trained from single frame, then the FC layers are fine-tuned from scratch.
    def __init__(self, num_class):
        super(conv_pooling, self).__init__()

        self.conv=nn.Sequential(* list(torchvision.models.resnet101().children())[:-2])
        self.time_pooling=nn.MaxPool3d(kernel_size=(120,1,1))
        self.average_pool=nn.AvgPool3d(kernel_size=(1,7,7))
        self.linear1=nn.Linear(2048,2048)
        self.linear2=nn.Linear(2048, num_class)
    def forward(self, x):
        t_len=x.size(2)
        conv_out_list=[]
        for i in range(t_len):
            conv_out_list.append(self.conv(torch.squeeze(x[:,:,i,:,:])))
        conv_out=self.time_pooling(torch.stack(conv_out_list,2))
        conv_out = self.average_pool(conv_out)
        conv_out=self.linear1(conv_out.view(conv_out.size(0),-1))
        conv_out=self.linear2(conv_out)
        return conv_out

class cnn_lstm(nn.Module):
    # Input size: 30x224x224
    # The CNN structure is first trained from single frame, then the lstm is fine-tuned from scratch.
    def __init__(self, num_class):
        super(cnn_lstm, self).__init__()

        self.conv = nn.Sequential(*list(torchvision.models.resnet101().children())[:-1])
        self.lstm = nn.LSTM(2048,512,5,batch_first=True)
        self.fc=nn.Linear(512,num_class)

    def forward(self, x):
        t_len = x.size(2)
        conv_out_list = []
        for i in range(t_len):
            conv_out_list.append(self.conv(torch.squeeze(x[:, :, i, :, :])))
        conv_out=torch.stack(conv_out_list,1)
        conv_out,hidden=self.lstm(conv_out.view(conv_out.size(0),conv_out.size(1),-1))
        lstm_out=[]
        for j in range (conv_out.size(1)):
            lstm_out.append(self.fc(torch.squeeze(conv_out[:,j,:])))
        return torch.stack(lstm_out,1),hidden