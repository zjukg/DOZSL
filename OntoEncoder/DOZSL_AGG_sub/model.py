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
    def __init__(self, rel_edges, num_rel, params=None):
        super(CapsuleBase, self).__init__(params)
        # self.edge_index = edge_index
        # self.edge_type = edge_type
        self.rel_edges = rel_edges
        # self.device = self.edge_index.device
        self.init_embed = get_param((self.p.num_ent, self.p.init_dim))
        self.init_rel = get_param((num_rel * 2, self.p.gcn_dim))
        self.pca = SparseInputLinear(self.p.init_dim, self.p.num_factors * self.p.gcn_dim)
        conv_ls = []
        for i in range(self.p.gcn_layer):
            conv = DisenLayer(self.p.init_dim, self.p.gcn_dim, num_rel,
                               act=self.act, params=self.p, head_num=self.p.head_num)
            self.add_module('conv_{}'.format(i), conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls
        if self.p.mi_train:
            if self.p.mi_method == 'club_b':
                num_dis = int((self.p.num_factors) * (self.p.num_factors - 1) / 2)
                # print(num_dis)
                self.mi_Discs = nn.ModuleList([CLUBSample(self.p.gcn_dim, self.p.gcn_dim, self.p.gcn_dim) for fac in range(num_dis)])
            elif self.p.mi_method == 'club_s':
                self.mi_Discs = nn.ModuleList([CLUBSample((fac + 1 ) * self.p.gcn_dim, self.p.gcn_dim, (fac + 1 ) * self.p.gcn_dim) for fac in range(self.p.num_factors - 1)])

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        self.rel_drop = nn.Dropout(0.1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def lld_bst(self, sub, rel, drop1, mode='train'):
        x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.gcn_dim) # [N K F]
        r = self.init_rel
        for conv in self.conv_ls:
            x, r = conv(x, r, self.rel_edges, mode) # N K F
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

    def mi_cal(self, sub_emb):
        def loss_dependence_hisc(zdata_trn, ncaps, nhidden):
            loss_dep = torch.zeros(1).cuda()
            hH = (-1/nhidden)*torch.ones(nhidden, nhidden).cuda() + torch.eye(nhidden).cuda()
            kfactor = torch.zeros(ncaps, nhidden, nhidden).cuda()

            for mm in range(ncaps):
                data_temp = zdata_trn[:, mm * nhidden:(mm + 1) * nhidden]
                kfactor[mm, :, :] = torch.mm(data_temp.t(), data_temp)

            for mm in range(ncaps):
                for mn in range(mm + 1, ncaps):
                    mat1 = torch.mm(hH, kfactor[mm, :, :])
                    mat2 = torch.mm(hH, kfactor[mn, :, :])
                    mat3 = torch.mm(mat1, mat2)
                    teststat = torch.trace(mat3)

                    loss_dep = loss_dep + teststat
            return loss_dep

        def loss_dependence_club_s(sub_emb):
            mi_loss = 0.
            for i in range(self.p.num_factors - 1):
                bnd = i + 1
                mi_loss += self.mi_Discs[i](sub_emb[:, :bnd * self.p.gcn_dim], sub_emb[:, bnd * self.p.gcn_dim : (bnd + 1) * self.p.gcn_dim])
            return mi_loss

        def loss_dependence_club_b(sub_emb):
            mi_loss = 0.
            cnt = 0
            for i in range(self.p.num_factors):
                for j in range( i + 1, self.p.num_factors):
                    mi_loss += self.mi_Discs[cnt](sub_emb[:, i * self.p.gcn_dim: (i + 1) * self.p.gcn_dim], sub_emb[:, j * self.p.gcn_dim: (j + 1) * self.p.gcn_dim])
                    cnt += 1
            return mi_loss
        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                   torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        if self.p.mi_method == 'club_s':
            mi_loss = loss_dependence_club_s(sub_emb)
        elif self.p.mi_method == 'club_b':
            mi_loss = loss_dependence_club_b(sub_emb)
        elif self.p.mi_method == 'hisc':
            mi_loss = loss_dependence_hisc(sub_emb, self.p.num_factors, self.p.gcn_dim)
        elif self.p.mi_method == "dist":
            cor = 0.
            for i in range(self.p.num_factors):
                for j in range(i + 1, self.p.num_factors):
                    cor += DistanceCorrelation(sub_emb[:, i * self.p.gcn_dim: (i + 1) * self.p.gcn_dim], sub_emb[:, j * self.p.gcn_dim: (j + 1) * self.p.gcn_dim])
            return cor
        else:
            raise NotImplementedError

        return mi_loss

    def forward_base(self, sub, rel, drop1, mode):
        if not self.p.no_enc:
            x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.gcn_dim) # [N K F]
            r = self.init_rel
            for conv in self.conv_ls:
                x, r = conv(x, r, self.rel_edges, mode) # N K F
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
        mi_loss = self.mi_cal(sub_emb)

        return sub_emb, rel_emb, x, mi_loss

    def test_base(self, sub, rel, drop1, mode):
        if not self.p.no_enc:
            x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.gcn_dim) # [N K F]
            r = self.init_rel
            for conv in self.conv_ls:
                x, r = conv(x, r, self.rel_edges, mode) # N K F
                x = drop1(x)
        else:
            x = self.init_embed.view(-1, self.p.num_factors, self.p.gcn_dim)
            r = self.init_rel
            x = drop1(x)
        sub_emb = torch.index_select(x, 0, sub)
        # rel_emb = torch.index_select(self.init_rel, 0, rel).repeat(1, self.p.num_factors)
        rel_emb = torch.index_select(self.init_rel, 0, rel)

        return sub_emb, rel_emb, x, 0.


class DOZSL_AGG_sub(CapsuleBase):
    def __init__(self, rel_edges, params=None):
        super(self.__class__, self).__init__(rel_edges, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)
        # self.rel_weight = self.conv_ls[-1].rel_weight
        self.all_ent_embeds = None
        gamma_init = torch.FloatTensor([self.p.init_gamma])
        if not self.p.fix_gamma:
            self.register_parameter('gamma', Parameter(gamma_init))

    def lld_best(self, sub, rel):
        return self.lld_bst(sub, rel, self.drop)

    def forward(self, sub, rel, mode='train'):
        if mode == 'train' and self.p.mi_train:
            sub_emb, rel_emb, all_ent, corr = self.forward_base(sub, rel, self.drop, mode)
            sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim)
        else:
            sub_emb, rel_emb, all_ent, corr = self.test_base(sub, rel, self.drop, mode)

        self.all_ent_embeds = all_ent

        sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.gcn_dim)


        # calculate the score
        sub_emb_selected = torch.zeros([sub_emb.size(0), sub_emb.size(-1)], dtype=torch.float).cuda()
        for i in range(sub_emb.size(0)):
            rel_index = rel[i]
            sub_emb_selected[i] = torch.index_select(sub_emb[i], 0, rel_index)

        obj_emb = sub_emb_selected + rel_emb

        obj_emb = obj_emb.repeat(1, self.p.num_factors)  # [B, KF]
        obj_emb = obj_emb.view(-1, self.p.num_factors, self.p.gcn_dim)  # [B, K, F]


        if self.p.gamma_method == 'norm':
            x2 = torch.sum(obj_emb * obj_emb, dim=-1)
            y2 = torch.sum(all_ent * all_ent, dim=-1)
            xy = torch.einsum('bkf,nkf->bkn', [obj_emb, all_ent])
            x = self.gamma - (x2.unsqueeze(2) + y2.t() -  2 * xy)

        x_selected = torch.zeros([x.size(0), x.size(-1)], dtype=torch.float).cuda()
        for i in range(x.size(0)):
            rel_index = rel[i]
            x_selected[i] = torch.index_select(x[i], 0, rel_index)


        if self.p.score_order == 'after':
            pred = torch.sigmoid(x_selected)
        return pred, corr

