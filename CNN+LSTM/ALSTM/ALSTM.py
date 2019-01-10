import torch
import torch.nn as nn
import torchvision

class lstm_cell(nn.Module):
    def __init__(self, input_num, hidden_num):
        super(lstm_cell, self).__init__()

        self.input_num = input_num
        self.hidden_num = hidden_num

        self.Wxi = nn.Linear(self.input_num, self.hidden_num, bias=True)
        self.Whi = nn.Linear(self.hidden_num, self.hidden_num, bias=False)
        self.Wxf = nn.Linear(self.input_num, self.hidden_num, bias=True)
        self.Whf = nn.Linear(self.hidden_num, self.hidden_num, bias=False)
        self.Wxc = nn.Linear(self.input_num, self.hidden_num, bias=True)
        self.Whc = nn.Linear(self.hidden_num, self.hidden_num, bias=False)
        self.Wxo = nn.Linear(self.input_num, self.hidden_num, bias=True)
        self.Who = nn.Linear(self.hidden_num, self.hidden_num, bias=False)

    def forward(self, xt, ht_1, ct_1):
        it = torch.sigmoid(self.Wxi(xt) + self.Whi(ht_1))
        ft = torch.sigmoid(self.Wxf(xt) + self.Whf(ht_1))
        ot = torch.sigmoid(self.Wxo(xt) + self.Who(ht_1))
        ct = ft * ct_1 + it * torch.tanh(self.Wxc(xt) + self.Whc(ht_1))
        ht = ot * torch.tanh(ct)
        return  ht, ct


class ALSTM(nn.Module):

    def __init__(self, input_num, hidden_num, num_layers,out_num ):
        super(ALSTM, self).__init__()

        # Make sure that `hidden_num` are lists having len == num_layers
        hidden_num = self._extend_for_multilayer(hidden_num, num_layers)
        if not len(hidden_num) == num_layers:
            raise ValueError('The length of hidden_num is not consistent with num_layers.')

        self.input_num = input_num
        self.hidden_num = hidden_num
        self.num_layers = num_layers
        self.out_num = out_num

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_num = self.input_num if i == 0 else self.hidden_num[i - 1]
            cell_list.append(lstm_cell(cur_input_num,self.hidden_num[i]))

        self.cell_list = nn.ModuleList(cell_list)
        self.conv=nn.Sequential(*list(torchvision.models.resnet101().children())[:-2])
        self.Wha=nn.Linear(self.hidden_num[-1],49)
        self.fc=nn.Linear(self.hidden_num[-1],self.out_num)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x, hidden_state=None):
        #input model: batch x channel x time x height x width
        #input size: 30 x 224 x 224

        # init -1 time hidden units
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=x.size(0))
        out_list=[]
        seq_len = x.size(2)

        for t in range(seq_len):
            output_t = []
            for layer_idx in range(self.num_layers):
                if 0==t:
                    ht_1, ct_1 = hidden_state[layer_idx][0],hidden_state[layer_idx][1]
                    attention_h=hidden_state[-1][0]
                else:
                    ht_1, ct_1 = hct_1[layer_idx][0],hct_1[layer_idx][1]
                if 0==layer_idx:
                    feature_map=self.conv(x[:, :, t, :, :])
                    feature_map=feature_map.view(feature_map.size(0),feature_map.size(1),-1)
                    attention_map=self.Wha(attention_h)
                    attention_map=torch.unsqueeze(self.softmax(attention_map),1)
                    attention_feature=attention_map*feature_map
                    attention_feature=torch.sum(attention_feature,2)
                    ht, ct = self.cell_list[layer_idx](attention_feature,ht_1, ct_1)
                    output_t.append([ht,ct])
                else:
                    ht, ct = self.cell_list[layer_idx](output_t[layer_idx-1][0], ht_1, ct_1)
                    output_t.append([ht,ct])
            attention_h=output_t[-1][0]
            hct_1=output_t
            out_list.append(self.fc(output_t[-1][0]))


        return torch.stack(out_list,1)


    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append([torch.zeros(batch_size, self.hidden_num[i]),torch.zeros(batch_size, self.hidden_num[i])])
        return init_states


    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

