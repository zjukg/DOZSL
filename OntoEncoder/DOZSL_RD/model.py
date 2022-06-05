from helper import *
import torch.nn as nn


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        return (-(mu - y_samples) ** 2 / 2. / logvar.exp()).sum(dim=1).mean(dim=0)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class BaseModel(torch.nn.Module):
    def __init__(self, params, device):
        super(BaseModel, self).__init__()

        self.p = params
        self.device = device
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
    def __init__(self, num_rel, params=None, device=None):
        super(CapsuleBase, self).__init__(params, device)

        self.init_embed = get_param((self.p.num_ent, self.p.init_dim))
        self.init_rel = get_param((num_rel, self.p.embed_dim))

        self.pca = SparseInputLinear(self.p.init_dim, self.p.num_factors * self.p.embed_dim)
        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        self.rel_drop = nn.Dropout(0.1)
        self.leakyrelu = nn.LeakyReLU(0.2)
    def forward_base(self, sub, rel, drop1):
        if not self.p.no_enc:
            x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.embed_dim)  # [N K F]
            r = self.init_rel

        else:
            x = self.init_embed
            r = self.init_rel
            x = drop1(x)
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(self.init_rel, 0, rel)



        return sub_emb, rel_emb, x

    def test_base(self, sub, rel, drop1):
        if not self.p.no_enc:
            x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.embed_dim)  # [N K F]
            r = self.init_rel

        else:
            x = self.init_embed.view(-1, self.p.num_factors, self.p.embed_dim)
            r = self.init_rel
            x = drop1(x)
        sub_emb = torch.index_select(x, 0, sub)
        # rel_emb = torch.index_select(self.init_rel, 0, rel).repeat(1, self.p.num_factors)
        rel_emb = torch.index_select(self.init_rel, 0, rel)



        return sub_emb, rel_emb, x


class DOZSL_Random(CapsuleBase):
    def __init__(self, params=None, device=None):
        super(self.__class__, self).__init__(params.num_rel, params, device)
        self.drop = torch.nn.Dropout(self.p.hid_drop)
        self.all_ent_embeds = None
        gamma_init = torch.FloatTensor([self.p.init_gamma])
        if not self.p.fix_gamma:
            self.register_parameter('gamma', Parameter(gamma_init))


    def forward(self, sub, rel, mode='train'):
        if mode == 'train':
            sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop)

            sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.embed_dim)
        else:
            sub_emb, rel_emb, all_ent = self.test_base(sub, rel, self.drop)


        self.all_ent_embeds = all_ent  # [N,K,F]
        sub_emb = sub_emb.view(-1, self.p.num_factors, self.p.embed_dim)  # [B,K,F]


        # calculate the score
        sub_emb_selected = torch.zeros([sub_emb.size(0), sub_emb.size(-1)], dtype=torch.float).cuda()
        for i in range(sub_emb.size(0)):
            rel_index = rel[i]
            sub_emb_selected[i] = torch.index_select(sub_emb[i], 0, rel_index)

        obj_emb = sub_emb_selected + rel_emb

        obj_emb = obj_emb.repeat(1, self.p.num_factors)  # [B, KF]
        obj_emb = obj_emb.view(-1, self.p.num_factors, self.p.embed_dim)  # [B, K, F]

        if self.p.gamma_method == 'norm':
            x2 = torch.sum(obj_emb * obj_emb, dim=-1)
            y2 = torch.sum(all_ent * all_ent, dim=-1)
            xy = torch.einsum('bkf,nkf->bkn', [obj_emb, all_ent])
            x = self.gamma - (x2.unsqueeze(2) + y2.t() - 2 * xy)  # [B,K,N]

        x_selected = torch.zeros([x.size(0), x.size(-1)], dtype=torch.float).cuda()
        for i in range(x.size(0)):
            rel_index = rel[i]
            x_selected[i] = torch.index_select(x[i], 0, rel_index)

        pred = torch.sigmoid(x_selected)
        return pred


