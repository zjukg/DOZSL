import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super(GraphConv, self).__init__()

        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        self.w = nn.Parameter(torch.empty(in_channels, out_channels))
        self.b = nn.Parameter(torch.zeros(out_channels))
        xavier_uniform_(self.w)

        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, inputs, adj):
        if self.dropout is not None:
            inputs = self.dropout(inputs)

        outputs = torch.mm(adj, torch.mm(inputs, self.w)) + self.b

        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs

class FN(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(FN, self).__init__()

        self.model = nn.Sequential(
            # nn.BatchNorm1d(input_dims),
                                   nn.Linear(in_features=input_dims, out_features=output_dims, bias=False))

    def forward(self, x):
        y = self.model(x)
        return y


class GCN(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_layers):
        super(GCN, self).__init__()


        # self.adj = adj

        hl = hidden_layers.split(',')
        if hl[-1] == 'd':
            dropout_last = True
            hl = hl[:-1]
        else:
            dropout_last = False

        i = 0
        layers = []
        last_c = in_channels  # dim.input
        for c in hl:
            if c[0] == 'd':
                dropout = True
                c = c[1:]
            else:
                dropout = False
            c = int(c)

            i += 1
            conv = GraphConv(last_c, c, dropout=dropout)
            self.add_module('conv{}'.format(i), conv)
            layers.append(conv)

            last_c = c

        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout_last)
        self.add_module('conv-last', conv)
        layers.append(conv)

        self.layers = layers

    def forward(self, x, adj):
        for conv in self.layers:
            x = conv(x, adj)


        return F.normalize(x)




