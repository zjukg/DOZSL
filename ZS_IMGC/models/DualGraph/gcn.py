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


class AttentiveGCN(GCN):

    def __init__(self, in_channels, out_channels, hidden_layers):
        super(self.__class__, self).__init__(in_channels, out_channels, hidden_layers)




    def add_atten_cos(self, inputs):
        output = inputs
        # print output

        # consin distance
        output = F.normalize(output)
        output_T = output.t()
        logits = torch.mm(output, output_T)

        # mask = self.mask
        # mask /= torch.mean(mask)
        # print self.mask
        logits = logits * 300

        # logits = logits * self.mask.t()
        coefs = F.softmax(logits, dim=1)
        # coefs = F.softmax(logits, dim=1)
        # coefs = logits
        # print coefs
        coefs = torch.where(coefs < 1e-3, torch.full_like(coefs, 0), coefs)
        # print(coefs)
        output_atten = torch.mm(coefs, inputs)
        # print output_atten
        return output_atten, coefs


    def forward(self, x, adj):

        for conv in self.layers:
            x = conv(x, adj)

        x = F.normalize(x)

        x_atten, coefs = self.add_atten_cos(x)
        return x_atten, coefs
        # return x

