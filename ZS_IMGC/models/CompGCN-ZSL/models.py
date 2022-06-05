from layers import CompGCNConv
from utils import *




class BaseModel(torch.nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.p = params
        self.act = torch.nn.LeakyReLU(negative_slope=0.2)


class CompGCNBase(BaseModel):
    def __init__(self, edge_index, edge_type, params=None):
        super(CompGCNBase, self).__init__(params)

        self.edge_index = edge_index
        self.edge_type = edge_type
        self.init_embed = get_param((self.p.num_ent, self.p.input_dim))
        self.device = self.edge_index.device
        self.drop = torch.nn.Dropout(self.p.hid_drop)

        self.init_rel = get_param((self.p.num_rel, self.p.input_dim))

        self.conv1 = CompGCNConv(self.p.input_dim, self.p.hidden_dim, self.p.num_rel, act=self.act, params=self.p)

        self.conv2 = CompGCNConv(self.p.hidden_dim, self.p.output_dim, self.p.num_rel, act=None,
                                 params=self.p)

    def forward(self):
        r = torch.cat([self.init_rel, -self.init_rel], dim=0)

        x, r = self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
        x = self.drop(x)
        x, r = self.conv2(x, self.edge_index, self.edge_type, rel_embed=r)
        x = self.drop(x)

        return F.normalize(x)

