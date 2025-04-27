import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
import networkx as nx
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn import GCNConv
import argparse

def get_tp_args():
    """
    Argument parser for running link_pred_check.py in command line
    """
    parser = argparse.ArgumentParser('Tie Predictability')
    parser.add_argument('--type', type=str, help='data used', default='t1_pos')
    return parser.parse_args()

def set_tp_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class SimpleGCN(torch.nn.Module):
    def __init__(self,in_dim, z_dim, h_dim=32):
        super().__init__()
        self.z_dim = z_dim
        self.training = True
        self.conv1 = GCNConv(in_dim, h_dim, add_self_loops=True)
        self.conv2 = GCNConv(h_dim, self.z_dim, add_self_loops=True)
        self.ll_y = torch.nn.Linear(self.z_dim + 2, 1)
        self.act = torch.nn.LeakyReLU()
    def forward(self, x, edge_index):
        x_1 = self.conv1(x, edge_index)
        x_1 = x_1.relu()
        x_1 = F.dropout(x_1, p=0.5, training=self.training)
        emb_out = self.act(self.conv2(x_1, edge_index))
        return emb_out

class SallowEmb(torch.nn.Module):
    def __init__(self, node_size, z_dim):
        super().__init__()
        # Structural Encoder (Separate F+N)
        self.N = node_size
        self.training = True
        self.embedding = torch.nn.Embedding(node_size, z_dim)
        self.act = torch.nn.LeakyReLU()
    def forward(self, all_id):
        emb_out = self.act(self.embedding(all_id))
        return emb_out

class InnerProductDecoder(torch.nn.Module):
    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value
    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj

def recon_loss(z, edge_tar, neg_edge_tar):
    EPS = 1e-15
    decoder = InnerProductDecoder()
    pos_edge_index, neg_edge_index = edge_tar, neg_edge_tar
    pos_loss = -torch.log(decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
    neg_loss = -torch.log(1 - decoder(z, neg_edge_index, sigmoid=True) +
                          EPS).mean()
    return pos_loss + neg_loss

def evaluate(z, pos_edge_index, neg_edge_index):
    pos_y = z.new_ones(pos_edge_index.size(1))
    neg_y = z.new_zeros(neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0)
    decoder = InnerProductDecoder()
    pos_pred = decoder(z, pos_edge_index, sigmoid=True)
    neg_pred = decoder(z, neg_edge_index, sigmoid=True)
    pred = torch.cat([pos_pred, neg_pred], dim=0)
    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
    return roc_auc_score(y, pred), average_precision_score(y, pred)


def train_FullGraph():
    edge_index = np.load('Dataset/twi22/'+data[:2]+'/'+data+'_edge.npy')    # (num_edge, 2)
    bot_label = np.load('Dataset/twi22/'+data[:2]+'/'+data+'_bot_label.npy')

    botData = Data(edge_index=torch.LongTensor(edge_index).t().contiguous())
    transform = RandomLinkSplit(is_undirected=False, num_val=0, num_test=0.3, neg_sampling_ratio=1.0)
    train_edge, _, test_edge = transform(botData)

    HHtest_edge_select, BHtest_edge_select = [], []
    for i in range(len(test_edge.edge_index[0])):
        follower_id = test_edge.edge_index[0][i].item()
        followed_id = test_edge.edge_index[1][i].item()
        if bot_label[follower_id] == 0 and bot_label[followed_id] == 0:
            HHtest_edge_select.append((follower_id, followed_id))
        elif (bot_label[follower_id] == 1 and bot_label[followed_id] == 0) or (bot_label[follower_id] == 0 and bot_label[followed_id] == 1):
            BHtest_edge_select.append((follower_id, followed_id))


    edgeH_pos, edgeB_pos = [], []
    for (e0, e1) in HHtest_edge_select:
        edgeH_pos.append([e0, e1])
    for (e0, e1) in BHtest_edge_select:
        edgeB_pos.append([e0, e1])

    edgeH_pos, edgeB_pos = torch.LongTensor(edgeH_pos).t(), torch.LongTensor(edgeB_pos).t()
    #print(train_edge.edge_index.shape, edgeH_pos.shape, edgeB_pos.shape)
    all_node_id = torch.LongTensor([i for i in range(len(bot_label))])

    #model = SimpleGCN(in_dim=1, z_dim=32)
    model = SallowEmb(node_size=len(bot_label), z_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(100):
        model.train()
        model.training = True
        optimizer.zero_grad()
        pos_edge, neg_edge = train_edge.edge_label_index[:, train_edge.edge_label.nonzero()].squeeze(-1), \
                             train_edge.edge_label_index[:, (1 - train_edge.edge_label).nonzero()].squeeze(-1)
        #emb_out = model(botData.x, pos_edge)  # semi-supervised GCN
        emb_out = model(all_node_id)    # shallow model
        loss = recon_loss(emb_out, pos_edge, neg_edge)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            model.training = False
            #embH, embB = model(botData.x, edgeH_pos), model(botData.x, edgeB_pos)   # semi-supervised GCN
            embH, embB = model(all_node_id), model(all_node_id)  # shallow model

            num_Hedge, num_Bedge = len(edgeH_pos[0]), len(edgeB_pos[0])
            y_posH = embH.new_ones(edgeH_pos.size(1))
            y_posB = embB.new_ones(edgeB_pos.size(1))

            neg_edge_test = test_edge.edge_label_index[:, (1 - test_edge.edge_label).nonzero()].squeeze(-1)
            edgeH_neg, edgeB_neg = neg_edge_test[:,:num_Hedge], neg_edge_test[:,:num_Bedge]


            y_negH, y_negB = edgeH_neg.new_zeros(edgeH_neg.size(1)), edgeB_neg.new_zeros(edgeB_neg.size(1))

            yH, yB = torch.cat([y_posH, y_negH], dim=0), torch.cat([y_posB, y_negB], dim=0)

            decoder = InnerProductDecoder()
            pred_posH, pred_posB = decoder(embH, edgeH_pos, sigmoid=True), decoder(embB, edgeB_pos, sigmoid=True)
            neg_predH, neg_predB = decoder(embH, edgeH_neg, sigmoid=True), decoder(embB, edgeB_neg, sigmoid=True)

            print("Epoch: ", epoch, " Test data: ", data)
            print("Average Human/Bot link: {:.4f} {:.4f}".format(np.mean(pred_posH.detach().numpy()), np.mean(pred_posB.detach().numpy())))

            predH, predB = torch.cat([pred_posH, neg_predH], dim=0).detach().cpu().numpy(), \
                                   torch.cat([pred_posB, neg_predB], dim=0).detach().cpu().numpy()
            print("Human: {:.4f} {:.4f}".format(roc_auc_score(yH, predH), average_precision_score(yH, predH)))
            print("Bot: {:.4f} {:.4f}".format(roc_auc_score(yB, predB), average_precision_score(yB, predB)))
            print("-------------------------")




def train_InfEgoGraph():
    edge_index = np.load('Dataset/twi22/'+data[:2]+'/'+data+'_edge.npy')    # (num_edge, 2)
    bot_label = np.load('Dataset/twi22/'+data[:2]+'/'+data+'_bot_label.npy')
    outcome = np.load('Dataset/twi22/'+data[:2]+'/'+data+'_y.npy')
    prop_label = np.load('Dataset/twi22/'+data[:2]+'/'+data+'_prop_label.npy')

    botData = Data(x=torch.FloatTensor(outcome).unsqueeze(-1), edge_index=torch.LongTensor(edge_index).t().contiguous())
    transform = RandomLinkSplit(is_undirected=False, num_val=0, num_test=0.3, neg_sampling_ratio=1.0)
    train_edge, _, test_edge = transform(botData)

    test_edge_select = []
    for i in range(len(test_edge.edge_index[0])):
        test_edge_select.append((test_edge.edge_index[0][i].item(), test_edge.edge_index[1][i].item()))

    G = nx.DiGraph()
    for e in range(len(edge_index)):
        G.add_edge(edge_index[e][0], edge_index[e][1])
    G = G.reverse() # for checking precessor

    edgeH_pos, edgeB_pos = [], []
    for node in G.nodes:
        if prop_label[node] == 1:
            if bot_label[node] == 0:
                ego_graph = nx.ego_graph(G, node)
                ego_graph = ego_graph.reverse()
                edgeH_pos.extend(list(ego_graph.edges()))
            elif bot_label[node] == 1:
                ego_graph = nx.ego_graph(G, node)
                ego_graph = ego_graph.reverse()
                edgeB_pos.extend(list(ego_graph.edges()))

    #print(len(test_edge_select), len(edgeH_pos), len(edgeB_pos))
    human_ego_test = set(test_edge_select)&set(edgeH_pos) # remove edge in the train set
    bot_ego_test = set(test_edge_select)&set(edgeB_pos)  # remove edge in the train set
    #print(len(human_ego_test), len(bot_ego_test))

    edgeH_pos, edgeB_pos = [], []
    for (e0, e1) in human_ego_test:
        edgeH_pos.append([e0, e1])
    for (e0, e1) in bot_ego_test:
        edgeB_pos.append([e0, e1])

    edgeH_pos, edgeB_pos = torch.LongTensor(edgeH_pos).t(), torch.LongTensor(edgeB_pos).t()
    #print(train_edge.edge_index.shape, edgeH_pos.shape, edgeB_pos.shape)
    all_node_id = torch.LongTensor([i for i in range(len(bot_label))])

    model = SimpleGCN(in_dim=1, z_dim=32)
    #model = SallowEmb(node_size=len(bot_label), z_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(100):
        model.train()
        model.training = True
        optimizer.zero_grad()
        pos_edge, neg_edge = train_edge.edge_label_index[:, train_edge.edge_label.nonzero()].squeeze(-1), \
                             train_edge.edge_label_index[:, (1 - train_edge.edge_label).nonzero()].squeeze(-1)
        emb_out = model(botData.x, pos_edge)  # semi-supervised GCN
        #emb_out = model(all_node_id)    # shallow model
        loss = recon_loss(emb_out, pos_edge, neg_edge)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            model.training = False
            embH, embB = model(botData.x, edgeH_pos), model(botData.x, edgeB_pos)   # semi-supervised GCN
            #embH, embB = model(all_node_id), model(all_node_id)  # shallow model

            num_Hedge, num_Bedge = len(edgeH_pos[0]), len(edgeB_pos[0])
            y_posH = embH.new_ones(edgeH_pos.size(1))
            y_posB = embB.new_ones(edgeB_pos.size(1))

            neg_edge_test = test_edge.edge_label_index[:, (1 - test_edge.edge_label).nonzero()].squeeze(-1)
            edgeH_neg, edgeB_neg = neg_edge_test[:,:num_Hedge], neg_edge_test[:,:num_Bedge]


            y_negH, y_negB = edgeH_neg.new_zeros(edgeH_neg.size(1)), edgeB_neg.new_zeros(edgeB_neg.size(1))

            yH, yB = torch.cat([y_posH, y_negH], dim=0), torch.cat([y_posB, y_negB], dim=0)

            decoder = InnerProductDecoder()
            pred_posH, pred_posB = decoder(embH, edgeH_pos, sigmoid=True), decoder(embB, edgeB_pos, sigmoid=True)
            neg_predH, neg_predB = decoder(embH, edgeH_neg, sigmoid=True), decoder(embB, edgeB_neg, sigmoid=True)

            print("Epoch: ", epoch, " Test data: ", data)
            print("Average Human/Bot link: {:.4f} {:.4f}".format(np.mean(pred_posH.detach().numpy()), np.mean(pred_posB.detach().numpy())))

            predH, predB = torch.cat([pred_posH, neg_predH], dim=0).detach().cpu().numpy(), \
                                   torch.cat([pred_posB, neg_predB], dim=0).detach().cpu().numpy()
            print("Human: {:.4f} {:.4f}".format(roc_auc_score(yH, predH), average_precision_score(yH, predH)))
            print("Bot: {:.4f} {:.4f}".format(roc_auc_score(yB, predB), average_precision_score(yB, predB)))
            print("-------------------------")


if __name__ == "__main__":
    args = get_tp_args()
    data = args.type
    set_tp_seed(101)
    #train_InfEgoGraph()
    train_FullGraph()