from helper import *
import torch.nn as nn
from DisenLayer import *

class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
#         print(x_samples.size())
#         print(y_samples.size())
        mu, logvar = self.get_mu_logvar(x_samples)

        return (-(mu - y_samples)**2 /2./logvar.exp()).sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

class BaseModel(torch.nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.p = params
        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):  # *nn.Linear* does not accept sparse *x*.
        return torch.mm(x, self.weight) + self.bias


class CapsuleBase(BaseModel):
    def __init__(self, edge_index, edge_type, num_rel, params=None):
        super(CapsuleBase, self).__init__(params)
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.device = self.edge_index.device
        self.init_embed = get_param((self.p.num_ent, self.p.init_dim))
        self.init_rel = get_param((num_rel * 2, self.p.gcn_dim))
        self.pca = SparseInputLinear(self.p.init_dim, self.p.num_factors * self.p.gcn_dim)
        conv_ls = []
        for i in range(self.p.gcn_layer):
            conv = DisenLayer(self.edge_index, self.edge_type, self.p.init_dim, self.p.gcn_dim, num_rel,
                               act=self.act, params=self.p, head_num=self.p.head_num)
            self.add_module('conv_{}'.format(i), conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        self.rel_drop = nn.Dropout(0.1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def lld_bst(self, sub, rel, drop1, mode='train'):
        x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.gcn_dim) # [N K F]
        r = self.init_rel
        for conv in self.conv_ls:
            x, r = conv(x, r, mode) # N K F
            if self.p.mi_drop:
                x = drop1(x)
            else:
                continue

        sub_emb = torch.index_select(x, 0, sub)
        lld_loss = 0.
        sub_emb = sub_emb.view(-1, self.p.gcn_dim * self.p.num_factors)
        if self.p.mi_method == 'club_s':
            for i in range(self.p.num_factors - 1):
                bnd = i + 1
                lld_loss += self.mi_Discs[i].learning_loss(sub_emb[:, :bnd * self.p.gcn_dim], sub_emb[:, bnd * self.p.gcn_dim : (bnd + 1) * self.p.gcn_dim])
        
        elif self.p.mi_method == 'club_b':
            cnt = 0
            for i in range(self.p.num_factors):
                for j in range( i + 1, self.p.num_factors):
                    lld_loss += self.mi_Discs[cnt].learning_loss(sub_emb[:, i * self.p.gcn_dim: (i + 1) * self.p.gcn_dim], sub_emb[:, j * self.p.gcn_dim: (j + 1) * self.p.gcn_dim])
                    cnt += 1
        return lld_loss



    def forward_base(self, sub, rel, drop1, mode):
        if not self.p.no_enc:
            x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.gcn_dim) # [N K F]
            r = self.init_rel
            for conv in self.conv_ls:
                x, r = conv(x, r, mode) # N K F
                x = drop1(x)
        else:
            x = self.init_embed
            r = self.init_rel
            x = drop1(x)
        sub_emb = torch.index_select(x, 0, sub)
        # rel_emb = torch.index_select(self.init_rel, 0, rel).repeat(1, self.p.num_factors)
        rel_emb = torch.index_select(self.init_rel, 0, rel)

        mi_loss = 0.
        sub_emb = sub_emb.view(-1, self.p.gcn_dim * self.p.num_factors)
        # mi_loss = self.mi_cal(sub_emb)

        return sub_emb, rel_emb, x, mi_loss

    def test_base(self, sub, rel, drop1, mode):
        if not self.p.no_enc:
            x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.gcn_dim) # [N K F]
            r = self.init_rel
            for conv in self.conv_ls:
                x, r = conv(x, r, mode) # N K F
                x = drop1(x)
        else:
            x = self.init_embed.view(-1, self.p.num_factors, self.p.gcn_dim)
            r = self.init_rel
            x = drop1(x)
        sub_emb = torch.index_select(x, 0, sub)
        # rel_emb = torch.index_select(self.init_rel, 0, rel).repeat(1, self.p.num_factors)
        rel_emb = torch.index_select(self.init_rel, 0, rel)

        return sub_emb, rel_emb, x, 0.


class RGAT(CapsuleBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)
        # self.rel_weight = self.conv_ls[-1].rel_weight
        self.all_ent_embeds = None
        gamma_init = torch.FloatTensor([self.p.init_gamma])
        if not self.p.fix_gamma:
            self.register_parameter('gamma', Parameter(gamma_init))

    def lld_best(self, sub, rel):
        return self.lld_bst(sub, rel, self.drop)

    def forward(self, sub, rel, mode='train'):

        sub_emb, rel_emb, all_ent, corr = self.test_base(sub, rel, self.drop, mode)



        sub_emb = sub_emb.view(-1, self.p.gcn_dim)  # [B, F]
        rel_emb = rel_emb.view(-1, self.p.gcn_dim)  # [B, F]
        all_ent = all_ent.view(-1, self.p.gcn_dim)  # [N, F]


        self.all_ent_embeds = all_ent

        # calculate the score
        obj_emb = sub_emb + rel_emb


        if self.p.gamma_method == 'norm':
            x2 = torch.sum(obj_emb * obj_emb, dim=-1)
            y2 = torch.sum(all_ent * all_ent, dim=-1)
            xy = torch.einsum('bf,nf->bn', [obj_emb, all_ent])
            x = self.gamma - (x2.unsqueeze(1) + y2.t() - 2 * xy)



        if self.p.score_order == 'after':
            pred = torch.sigmoid(x)
        return pred, corr

