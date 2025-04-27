import torch
import os
import numpy as np
import pandas as pd
from AdvG import InnerProductDecoder

def recon_loss(z, edge_tar, neg_edge_tar):
    EPS = 1e-15
    decoder = InnerProductDecoder()
    pos_edge_index, neg_edge_index = edge_tar, neg_edge_tar
    pos_loss = -torch.log(decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
    neg_loss = -torch.log(1 - decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
    return pos_loss + neg_loss

def generate_counterfactual_edge2(follower, following, inv_edge_index, device):
    selected_edge = torch.cat([torch.LongTensor(follower).unsqueeze(-1),torch.LongTensor(following).unsqueeze(-1)],dim=-1)
    new_edge_index = torch.cat([inv_edge_index, selected_edge.T.to(device)], dim=1)
    return new_edge_index


def match_node(cf_graph, bot_label, prop_label, treat_idx, control_idx):
    '''
    judging whether the treated node in the original graph have its (controlled group) counterpart in the counterfactual graph

    '''
    friend_dict_cf = {}
    for i in range(len(cf_graph[0])):
        u, v = cf_graph[0][i].item(), cf_graph[1][i].item()
        friend_dict_cf.setdefault(u, []).append(v)

    treat_idx_ok, control_idx_ok = [], []
    for id in treat_idx.tolist(): # treat_idx: nodes follow bot inf in G
        if id not in friend_dict_cf:
            continue
        friend = friend_dict_cf[id]  # id's friend
        bot_prop_label = prop_label[friend] * bot_label[friend] # only bot+prop==1
        if (prop_label[friend].sum()>0) and (bot_prop_label.sum() == 0): # follow human inf (not bot inf) in Gcf
            treat_idx_ok.append(id)

    for id in control_idx.tolist(): # control_idx: nodes follow human inf in G
        if id not in friend_dict_cf:
            continue
        friend = friend_dict_cf[id]  # id's friend
        bot_prop_label = prop_label[friend] * bot_label[friend]  # only bot+prop==1
        if (bot_prop_label.sum()>0) and (prop_label[friend]-bot_prop_label).sum()==0: # follow bot inf (no human inf) in Gcf
            control_idx_ok.append(id)
    return torch.LongTensor(treat_idx_ok), torch.LongTensor(control_idx_ok)

def pairwise_similarity(matrix1, matrix2):
    EPS = 1e-15
    norm1 = torch.norm(matrix1, p=2, dim=1, keepdim=True)
    norm2 = torch.norm(matrix2, p=2, dim=1, keepdim=True)
    normalized_matrix1 = matrix1 / (norm1 + EPS)
    normalized_matrix2 = matrix2 / (norm2 + EPS)

    # calculate pair-wise similarities
    # similarity_matrix = torch.mm(normalized_matrix1, normalized_matrix2.t())
    similarity_matrix = torch.cdist(normalized_matrix1, normalized_matrix2, p=2)
    return similarity_matrix

def similarity_check(matrix1, matrix2):
    norm1 = torch.norm(matrix1, p=2, dim=1, keepdim=True)
    norm2 = torch.norm(matrix2, p=2, dim=1, keepdim=True)
    normalized_matrix1 = matrix1 / norm1
    normalized_matrix2 = matrix2 / norm2
    dist = torch.cdist(normalized_matrix1, normalized_matrix2, p=2).mean().item()
    cosim = torch.mm(normalized_matrix1, normalized_matrix2.t()).mean().item()
    print('Sim_dist: {:.4f}, Sim_cos: {:.4f}'.format(dist * 1000, cosim))


def create_dir(dirname):
    """
    Function to create a directory if it does not already exist.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)

class SaveStoppingUtil:
    """
    A general function for stopping the training and model saving
    Based on the validation loss for early stopping or other criterion

    """
    def __init__(self, args):

        # Some Utilities
        self.best_min_score = np.Inf   # i.e., MSE
        self.best_max_score = -np.Inf  # i.e., F1-score
        # patience (adapted from https://github.com/Bjarten/early-stopping-pytorch)
        self.patience_counter = 0
        self.patience_step = 5 # default
        self.early_stop = False
        self.Train_record = {'Epoch': [], 'aveT': [], 'aveC': [], 'bInf-hInf': [],
                             'eATE': [], 'ePEHE': [], 'tr_lcff': [], 'tr_rjf': [], 'va_rjf': []}
        self.gold_epoch = 0
        self.Model_dict_gold, self.Result_dict_gold = None, None
        self.Model_dict_mid, self.Result_dict_mid = None, None

    def record_train_msg(self, epoch, ave_treat, ave_control, treat_eff, eATE_test, ePEHE_test, r_cffool, r_jf, loss_jf_va):
        self.Train_record['Epoch'].append(epoch)
        self.Train_record['aveT'].append(ave_treat)
        self.Train_record['aveC'].append(ave_control)
        self.Train_record['bInf-hInf'].append(treat_eff)
        self.Train_record['eATE'].append(eATE_test)
        self.Train_record['ePEHE'].append(ePEHE_test)
        self.Train_record['tr_lcff'].append(np.mean(r_cffool))
        self.Train_record['tr_rjf'].append(np.mean(r_jf))
        self.Train_record['va_rjf'].append(loss_jf_va.detach().cpu().numpy())

    def save_train_msg(self, save_path, file_name):
        create_dir(save_path)
        res_cvrg = pd.DataFrame(self.Train_record)
        res_cvrg.to_csv(save_path+file_name, index=False)

    def direct_model_save(self, model_state_dict, save_model_path, save_model_name):
        create_dir(save_model_path)
        torch.save(model_state_dict, save_model_path + save_model_name)

    def update_gold_eval(self, score_eval, epoch_eval, model_state_dict, result_dict, best_cri='min_best'):
        '''
        Store best score_eval model
        '''
        improve = self._judge_improve(score_eval, best_cri)
        if improve:
            self.best_score_min = score_eval
            self.Model_dict_gold = model_state_dict
            self.Result_dict_gold = result_dict
            self.gold_epoch = epoch_eval

    def get_intermid_eval(self, model_state_dict, result_dict):
        '''
        Get model for the intermid check
        '''
        self.Model_dict_mid = model_state_dict
        self.Result_dict_mid = result_dict

    def _judge_improve(self, score_eval, best_cri):
        if best_cri == 'min_best':
            if score_eval < self.best_min_score:
                improve = True
            else:
                improve = False
        elif best_cri == 'max_best':
            if score_eval > self.best_max_score:
                improve = True
            else:
                improve = False
        return improve






