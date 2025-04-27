import torch
from torch_geometric.utils import degree
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class HomoEdgeDetector(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super(HomoEdgeDetector, self).__init__()
        self.convM1 = GATConv(in_dim, h_dim)
        self.convM2 = GATConv(h_dim, out_dim)

    def forward(self, x, edge_index, mask_homo=0.6):
        # edge_index: e(2, num_edge)
        xM1 = F.leaky_relu(self.convM1(x, edge_index))
        #xM1 = F.dropout(xM1, p=0.5, training=self.training)
        xM2 = self.convM2(xM1, edge_index)  # xM2: (num_nodes, h_dim)
        value = (xM2[edge_index[0]] * xM2[edge_index[1]]).sum(dim=1) # (num_edges)
        # select homophily edges
        _, topk_homo = torch.topk(value, int(len(value)*mask_homo), largest=True)
        _, topk_hetero = torch.topk(value, int(len(value)*(1-mask_homo)), largest=False)
        return edge_index[:,topk_homo], edge_index[:,topk_hetero], xM2

class BotImpact(torch.nn.Module):
    def __init__(self, in_dim, h_e, h_o, out_dim=1, heads=1):
        super(BotImpact, self).__init__()
        self.convZ1 = GATConv(in_dim, h_e, heads)
        self.convZ2 = GATConv(h_e*heads, h_e, heads)

        self.yNetS = torch.nn.Sequential(torch.nn.Linear(h_e * heads, h_o), torch.nn.LeakyReLU())
        self.yNet1 = torch.nn.Sequential(torch.nn.Linear(h_o, h_o), torch.nn.LeakyReLU(), torch.nn.Linear(h_o, out_dim), torch.nn.LeakyReLU())
        self.yNet0 = torch.nn.Sequential(torch.nn.Linear(h_o, h_o), torch.nn.LeakyReLU(), torch.nn.Linear(h_o, out_dim), torch.nn.LeakyReLU())

        self.propenNet = torch.nn.Sequential(torch.nn.Linear(h_e * heads, 2))
    def forward(self, x, edge_index, fake_x, fake_edge_index, treat_idx, control_idx):
        # generate node embedding
        xZ1, xfZ1 = F.relu(self.convZ1(x, edge_index)), F.relu(self.convZ1(fake_x, fake_edge_index))
        xZ1, xfZ1 = F.dropout(xZ1, p=0.5, training=self.training), F.dropout(xfZ1, p=0.5, training=self.training)
        xZ2, xfZ2 = self.convZ2(xZ1, edge_index), self.convZ2(xfZ1, fake_edge_index)    # xZ2/xfZ2: (num_nodes, h_dim)
        # predict outcome
        y1, yc0 = self.yNet1(self.yNetS(xZ2[treat_idx])), self.yNet0(self.yNetS(xfZ2[treat_idx]))
        y0, yc1 = self.yNet0(self.yNetS(xZ2[control_idx])), self.yNet1(self.yNetS(xfZ2[control_idx]))
        # predict treatment
        tprob  = self.propenNet(xZ2)
        return y1.squeeze(-1), yc0.squeeze(-1), y0.squeeze(-1), yc1.squeeze(-1), xZ2, xfZ2, tprob.squeeze(-1)

class Discriminator(torch.nn.Module):
    def __init__(self, in_dim, h_dim, heads=1):
        super(Discriminator, self).__init__()
        self.balanceNet = torch.nn.Sequential(torch.nn.Linear(in_dim * heads, h_dim), torch.nn.LeakyReLU(),
                                              torch.nn.Linear(h_dim, 1), torch.nn.Sigmoid())
    def forward(self, xZ2, xfZ2):
        # judge the node is from factual & counterfactual graph
        fprob, fprob_f = self.balanceNet(xZ2), self.balanceNet(xfZ2)
        return fprob.squeeze(-1), fprob_f.squeeze(-1)

class InnerProductDecoder(torch.nn.Module):
    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value
    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj