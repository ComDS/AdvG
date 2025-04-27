import numpy as np
import pandas as pd
import random
import os
import torch
from torch_geometric.utils import degree
from torch_geometric.utils import to_networkx
import networkx as nx
import argparse


def get_simu_args():
    """
    Argument parser for running synthetic.py in command line
    """
    parser = argparse.ArgumentParser('Simulation')

    parser.add_argument('--type', type=str, help='simulation network type', default='random')

    parser.add_argument('--sample_user', type=int, help='number of users in a network', default=3000)
    parser.add_argument('--sample_bot', type=int, help='number of bots in a network', default=100)
    parser.add_argument('--betaZ', type=float, help='influence of latent trait', default=1)
    parser.add_argument('--betaT', type=float, help='influence of a human influencer', default=2)
    parser.add_argument('--betaB', type=float, help='influence of a bot influencer', default=1)
    parser.add_argument('--EPSILON', type=float, help='random disturbance', default=0.2)
    return parser.parse_args()

def set_simu_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def generate_network(z, bl, type):
    '''
    input: latent trait, bot label
    simulating different network types:
    type = 'random'   # bots randomly connect
    type = 'randomu'  # bots randomly connect users
    type = 'highdu'   # bots connect high-degree users
    type = 'highbc'   # bots connect high-betweeness-centrality users
    type = 'highcc'   # bots connect high-closeness-centrality users
    '''
    edge_idx = []
    if type == 'random':
        for i in range(N_SAMPLE):
            for j in range(i + 1, N_SAMPLE):
                # pairs are humans
                if (bl[i] == 0) and (bl[j] == 0):
                    p = 0.02 if z[i] == z[j] else 0.005 # default: similar-0.02, dissimilar-0.005
                    friend = np.random.binomial(1, p)
                    if friend == 1:
                        edge_idx.append([i, j])
                        edge_idx.append([j, i])
                # the pairs have bots
                else:
                    p = 0.01    # pro of bots being followed (default: 0.01)
                    friend = np.random.binomial(1, p)
                    if friend == 1: # default: bots will follow back the users when being followed
                        edge_idx.append([i, j])
                        edge_idx.append([j, i])

    elif type == 'randomu':
        for i in range(N_USER):
            for j in range(i + 1, N_USER):
                p = 0.02 if z[i] == z[j] else 0.005 # default: 0.02, 0.005
                friend = np.random.binomial(1, p)
                if friend == 1:
                    edge_idx.append([i, j])
                    edge_idx.append([j, i])
        for u in range(N_USER):
            for b in range(N_USER, N_SAMPLE):
                p = 0.01    # pro of bots being followed (default: 0.01)
                friend = np.random.binomial(1, p)
                if friend == 1: # default: bots will follow back the users when being followed
                    edge_idx.append([u, b])
                    edge_idx.append([b, u])

    elif type == 'highdu':
        for i in range(N_USER):
            for j in range(i + 1, N_USER):
                p = 0.02 if z[i] == z[j] else 0.005
                friend = np.random.binomial(1, p)
                if friend == 1:
                    edge_idx.append([i, j])
                    edge_idx.append([j, i])
        ndeg = degree(torch.LongTensor(edge_idx).T[0,:])
        max_deg = torch.max(ndeg).item()
        for u in range(N_USER):
            for b in range(N_USER, N_SAMPLE):
                p = 0.02*ndeg[u]/max_deg
                friend = np.random.binomial(1, p)
                friendbu = np.random.binomial(1, p*0.8) # some user follow bot
                if friend == 1:
                    edge_idx.append([b, u])
                if friendbu == 1:
                    edge_idx.append([u, b])


    elif type == 'highbc':
        edge_idx_nx = []
        for i in range(N_USER):
            for j in range(i + 1, N_USER):
                p = 0.02 if z[i] == z[j] else 0.005
                friend = np.random.binomial(1, p)
                if friend == 1:
                    edge_idx.append([i, j])
                    edge_idx.append([j, i])
                    edge_idx_nx.append((i, j))
                    edge_idx_nx.append((j, i))
        G = nx.Graph(edge_idx_nx)
        betweenness_centrality = nx.betweenness_centrality(G)
        max_bc = max(betweenness_centrality.values())
        for u in range(N_USER):
            for b in range(N_USER, N_SAMPLE):
                p = 0.02*betweenness_centrality[u]/max_bc
                friend = np.random.binomial(1, p)
                friendbu = np.random.binomial(1, p*0.8) # some user follow bot
                if friend == 1:
                    edge_idx.append([b, u])
                if friendbu == 1:
                    edge_idx.append([u, b])

    elif type == 'highcc':
        edge_idx_nx = []
        for i in range(N_USER):
            for j in range(i + 1, N_USER):
                p = 0.02 if z[i] == z[j] else 0.005
                friend = np.random.binomial(1, p)
                if friend == 1:
                    edge_idx.append([i, j])
                    edge_idx.append([j, i])
                    edge_idx_nx.append((i, j))
                    edge_idx_nx.append((j, i))
        G = nx.Graph(edge_idx_nx)
        betweenness_centrality = nx.closeness_centrality(G)
        max_bc = max(betweenness_centrality.values())
        for u in range(N_USER):
            for b in range(N_USER, N_SAMPLE):
                p = 0.02*betweenness_centrality[u]/max_bc
                friend = np.random.binomial(1, p)
                friendbu = np.random.binomial(1, p * 0.8)  # some user follow bot
                if friend == 1:
                    edge_idx.append([b, u])
                if friendbu == 1:
                    edge_idx.append([u, b])
    return np.array(edge_idx)



def cal_outcome(Z, edge_idx, human_propagator_id, bl, eps):
    # edge_idx: (num_edge, 2)
    friend_dict = {}
    for i in range(len(edge_idx)):
        u, v = edge_idx[i][0], edge_idx[i][1]
        friend_dict.setdefault(u, []).append(v)

    # for neighbor homophily testing
    trait_sum_h, trait_sum_b = [], []
    bot_ids, human_ids = np.nonzero(bl)[0], np.nonzero(1-bl)[0]
    for prop in human_propagator_id:
        friend_id = friend_dict[prop]
        trait_sum_h.extend(Z[list(set(friend_id)&set(human_ids))].tolist())
    for bot in bot_ids:
        friend_id = friend_dict[bot]
        trait_sum_b.extend(Z[list(set(friend_id)&set(human_ids))].tolist())
    #print("Ave h: ", np.mean(trait_sum_h), " Ave b: ", np.mean(trait_sum_b))

    y = np.zeros(N_SAMPLE)
    Di, Bi = np.zeros(N_SAMPLE), np.zeros(N_SAMPLE)# 0-1 vector (sample_user + sample_bot)
    T = np.zeros(N_SAMPLE) # treat-control label
    for i in range(N_SAMPLE):
        if i not in friend_dict:
            y[i] = args.betaZ * Z[i] + eps[i]
            continue
        friend = friend_dict[i] # i's friend id
        prop_u = set(friend)&set(human_propagator_id)
        prop_b = np.nonzero(bl[friend])[0]
        di = 1 if len(prop_u)>0 else 0
        bi = 1 if len(prop_b)>0 else 0
        if bl[i]==0 and di==1 and bi==0:
            T[i] = 1    # control (the node is a human and follows a human)
        elif bl[i]==0 and di==0 and bi==1:
            T[i] = -1   # treat (the node is a human and follows a bot)
        Di[i], Bi[i] = di, bi
        y[i] = args.betaZ * Z[i] + args.betaT * di + args.betaB * bi + eps[i]
    return y, Di, Bi, T


def gen_simulate_data(type, bot_label, seed):
    '''
    Output synthetic network data
    Data includes:
        edge_index
        bot_label
        T (treatment label)
        outcome
        propagator (influencer label)
        out_data (csv files)
    '''
    if type in ['random', 'randomu', 'highdu', 'highbc', 'highcc']:
        Zu = np.random.choice([0, 1], N_USER)
        Zb = np.ones(N_BOT)
        Z = np.concatenate([Zu, Zb])
        propagator = np.concatenate([np.zeros(N_USER), np.ones(N_BOT)], axis=0)
        human_propagator_id = random.sample(set(np.nonzero(Zu)[0]), N_BOT) # N_humanInfluencer = N_botInfluencer

    propagator[human_propagator_id] = 1 # human propagator label
    edge_index = generate_network(Z, bot_label, type)
    #print("Network density: ", len(edge_index) / (N * N))

    eps = np.random.normal(0, args.EPSILON, size=N_SAMPLE)
    outcome, Di, Bi, T = cal_outcome(Z, edge_index, human_propagator_id, bot_label, eps)
    
    print("Num treat/control: ", pd.Series(T).value_counts())
    
    out_data = pd.DataFrame({'bot_label': bot_label,
                             'propagator': propagator,
                             'Di': Di,
                             'Bi': Bi,
                             'treated': T,
                             'purchase': outcome})

    if not os.path.isdir('Dataset/synthetic/' + type + '/'):
        os.makedirs('Dataset/synthetic/' + type + '/')
    np.save('Dataset/synthetic/' + type + '/' + str(seed) + '_edge.npy', edge_index)
    np.save('Dataset/synthetic/' + type + '/' + str(seed) + '_bot_label.npy', bot_label)
    np.save('Dataset/synthetic/' + type + '/' + str(seed) + '_T_label.npy', T)
    np.save('Dataset/synthetic/' + type + '/' + str(seed) + '_y.npy', outcome)
    np.save('Dataset/synthetic/' + type + '/' + str(seed) + '_prop_label.npy', propagator)
    out_data.to_csv('Dataset/synthetic/' + type + '/' + str(seed) + '_bot.csv', index=False)


if __name__ == "__main__":
    # get arguments from the command line
    args = get_simu_args()
    N_USER, N_BOT = args.sample_user, args.sample_bot
    N_SAMPLE = N_USER + N_BOT
    for seed in range(0, 100): # make repeated experiments by altering the random seed
        set_simu_seed(seed)
        bot_label = np.array([0] * N_USER + [1] * N_BOT)
        gen_simulate_data(args.type, bot_label, seed)
        print("Finish generate: ", seed)
