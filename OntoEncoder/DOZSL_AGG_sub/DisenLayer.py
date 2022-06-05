from helper import *
# from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import torch.nn as nn
from message_passing import MessagePassing

class DisenLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_rels, act=lambda x: x, params=None, head_num=1):
        super(self.__class__, self).__init__()

        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        # self.device = None
        self.head_num = head_num
        self.num_rels = num_rels
        # params for init
        self.drop = torch.nn.Dropout(self.p.dropout)
        self.dropout = torch.nn.Dropout(0.3)
        self.bn = torch.nn.BatchNorm1d(self.p.num_factors * out_channels)
        if self.p.bias:
            self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

        # num_edges = self.edge_index.size(1) // 2
        # if self.device is None:
        #     self.device = self.edge_index.device

        # self.in_index, self.out_index = self.edge_index[:, :num_edges], self.edge_index[:, num_edges:]
        # self.in_type, self.out_type = self.edge_type[:num_edges], self.edge_type[num_edges:]


        self.loop_index = torch.stack([torch.arange(self.p.num_ent), torch.arange(self.p.num_ent)]).cuda()
        self.loop_type = torch.full((self.p.num_ent,), 2 * self.num_rels, dtype=torch.long).cuda()

        # self.loop_type = torch.full((self.p.num_ent,), 2 * self.num_rels, dtype=torch.long).to(self.device)
        num_ent = self.p.num_ent

        self.leakyrelu = nn.LeakyReLU(0.2)
        # if self.p.att_mode == 'cat_emb' or self.p.att_mode == 'cat_weight':
        #     self.att_weight = get_param((1, self.p.num_factors, 2 * out_channels))
        # else:
        #     self.att_weight = get_param((1, self.p.num_factors, out_channels))
        # self.rel_weight = get_param((2 * self.num_rels + 1, self.p.num_factors, out_channels))
        self.loop_rel = get_param((1, out_channels))
        self.w_rel = get_param((out_channels, out_channels))

    def forward(self, x, rel_embed, rel_edges, mode):
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)

        out = torch.zeros(x.size(), dtype=torch.float).cuda()
        for k in range(self.p.num_factors):
            x_selected = torch.zeros([x.size(0), x.size(-1)], dtype=torch.float).cuda()
            for i in range(x.size(0)):
                x_selected[i] = x[i][k]
            # self-loop
            # loop_type = torch.full((self.p.num_ent,), k, dtype=torch.long).cuda()
            edge_index = torch.LongTensor(rel_edges[k]).cuda().t()
            edge_type = torch.full((len(rel_edges[k]),), k, dtype=torch.long).cuda()

            edge_index = torch.cat([edge_index, self.loop_index], dim=1)
            edge_type = torch.cat([edge_type, self.loop_type])


            tmp_out = self.propagate('add', edge_index, x=x_selected, edge_type=edge_type, rel_embed=rel_embed)

            for i in range(x.size(0)):
                out[i][k] = tmp_out[i]

        # out = self.propagate(edge_index, size=None, x=x, edge_type=edge_type, rel_embed=rel_embed, rel_weight=self.rel_weight)
        if self.p.bias:
            out = out + self.bias
        out = self.bn(out.view(-1, self.p.num_factors * self.p.gcn_dim)).view(-1, self.p.num_factors, self.p.gcn_dim)

        entity1 = out if self.p.no_act else self.act(out)

        return entity1, torch.matmul(rel_embed, self.w_rel)[:-1]

    def message(self, x_j, edge_type, rel_embed):
        '''
        edge_index_i : [E]
        x_i: [E, F]
        x_j: [E, F]
        '''
        rel_embed = torch.index_select(rel_embed, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_embed)

        return xj_rel

    def update(self, aggr_out):
        return aggr_out



    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'none':
            trans_embed = ent_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError

        return trans_embed

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)


def conj(a):
	a[..., 1] = -a[..., 1]
	return a
def com_mult(a, b):
	r1, i1 = a[..., 0], a[..., 1]
	r2, i2 = b[..., 0], b[..., 1]
	return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)
def ccorr(a, b):
	return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))