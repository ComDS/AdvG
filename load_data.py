import torch
import numpy as np
from torch_geometric.data import Data

'''
Training Data Elements

* edge_index (num_edge, 2): Social Relationships
* bot_label: 1=bot, 0=human
* treat_indicator: 1=control Gp, -1=treatment Gp, 0=ohters
* outcome: Attitude Score
* prop_label: Whether attitudinal propagator (for node matching)
'''

def construct_data_elements(edge_index, outcome, bot_label, treat_indicator):
    # cal basic
    N = len(outcome)  # num of nodes
    x = torch.cat([torch.FloatTensor(bot_label).unsqueeze(-1), torch.FloatTensor(treat_indicator).unsqueeze(-1)], dim=-1)
    # target: bot&human, attitude, treat&control
    target_var = torch.tensor(
        np.concatenate([bot_label[:, np.newaxis], outcome[:, np.newaxis], treat_indicator[:, np.newaxis]], axis=-1))  # (num_nodes, 3)
    botData = Data(x=x, edge_index=edge_index.t().contiguous(), y=target_var)
    return botData, N


def load_simulation_data(type, data_id):
    '''
    Loading Simulation Data
    Type: 'random', 'randomu', 'highbc', 'highcc', 'highdu'...
    '''
    # load data (repeat experiment in simulations)
    edge_index = torch.LongTensor(np.load('Dataset/synthetic/' + type + '/' + str(data_id) + '_edge.npy'))  # (num_edge, 2)
    bot_label = np.load('Dataset/synthetic/' + type + '/' + str(data_id) + '_bot_label.npy')
    treat_indicator = np.load('Dataset/synthetic/' + type + '/' + str(data_id) + '_T_label.npy')
    outcome = np.load('Dataset/synthetic/' + type + '/' + str(data_id) + '_y.npy')
    prop_label = np.load('Dataset/synthetic/' + type + '/' + str(data_id) + '_prop_label.npy')
    botData, N = construct_data_elements(edge_index, outcome, bot_label, treat_indicator)
    return botData, N, prop_label


def load_empirical_data(type):
    '''
    Loading Empirical Data
    Type: 't1_pos', 't2_pos', 't3_pos', 't1_neg', 't2_neg', 't3_neg'
    '''
    # load data (empirical settings)
    edge_index = torch.LongTensor(np.load('Dataset/twi22/'+type[:2]+'/'+type+'_edge.npy'))    # (num_edge, 2)
    bot_label = np.load('Dataset/twi22/'+type[:2]+'/'+type+'_bot_label.npy')
    treat_indicator = np.load('Dataset/twi22/'+type[:2]+'/'+type+'_T_label.npy')
    outcome = np.load('Dataset/twi22/'+type[:2]+'/'+type+'_y.npy')
    prop_label = np.load('Dataset/twi22/'+type[:2]+'/'+type+'_prop_label.npy')
    botData, N = construct_data_elements(edge_index, outcome, bot_label, treat_indicator)
    # prop_label: used for node matching
    return botData, N, prop_label