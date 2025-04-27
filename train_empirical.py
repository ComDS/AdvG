import numpy as np
import pandas as pd
import random
import warnings
import argparse

from utils import *
from AdvG import *
from evaluate import *
from load_data import load_simulation_data, load_empirical_data
from torch_geometric.data import Data


warnings.filterwarnings("ignore")

def get_train_args():
    """
    Argument parser for running train.py in command line
    """
    parser = argparse.ArgumentParser('Training AdvG for Empirical Data')

    # data parameters
    parser.add_argument('--type', type=str, help='data used', default='t1_pos')

    # for the empirical data, set '--effect_true'==0 (no pre-specified effect)
    # for the synthetic data, the value depends on the simulation settings (e.g., beta_b-beta_h = -1)
    parser.add_argument('--effect_true', type=float, help='ground-truth effect', default=0)

    # model parameters
    parser.add_argument('--mask_homo', type=float, help='mask edge percentage', default=0.8)
    parser.add_argument('--hm', type=int, help='hidden dimension of the homophily edge detector', default=32)
    parser.add_argument('--dzm', type=int, help='output dimension of the homophily edge detector', default=16)
    parser.add_argument('--hd', type=int, help='dimension of the discriminator', default=32)
    parser.add_argument('--he', type=int, help='dimension of the graph encoder', default=64)
    parser.add_argument('--ho', type=int, help='hidden dimension of the outcome generator', default=64)

    # training parameters
    parser.add_argument('--gpu', type=int, help='gpu', default=0)
    parser.add_argument('--ly', type=float, help='reg for outcome pred', default=1)
    parser.add_argument('--ljt', type=float, help='reg for treat pred', default=0.1)
    parser.add_argument('--ljg', type=float, help='reg for cf generate', default=100)
    parser.add_argument('--ljd', type=float, help='reg for cf discrim', default=1)

    parser.add_argument('--train_prec', type=float, help='precentage of training samples', default=0.5)
    parser.add_argument('--lr_mf', type=float, help='learning rate of model f', default=0.001)
    parser.add_argument('--lr_mg', type=float, help='learning rate of model g', default=0.001)
    parser.add_argument('--lr_md', type=float, help='learning rate of model d', default=0.001)
    parser.add_argument('--max_epoch', type=int, help='max num epochs', default=200)
    parser.add_argument('--eval_per_epoch', type=int, help='conduct evaluation per epoch', default=10)

    # saving embedding & intermediate result
    parser.add_argument('--check_emp_with_Naive', type=bool, help='comparsion with naive approach', default=True)
    parser.add_argument('--converge_check', type=bool, help='save training loss results', default=True)
    parser.add_argument('--save_emp_epoch_statedict', type=bool, help='save model on emp', default=False)
    parser.add_argument('--rep_epoch_check', type=int, help='save epoch result', default=180)

    # path dir
    parser.add_argument('--simu_rep_res_dir', type=str, help='simulation result store path', default='result/simu/')
    parser.add_argument('--emp_model_save_dir', type=str, help='empirical model store path', default='result/emp_model_save/')
    parser.add_argument('--conv_check_dir', type=str, help='result for training msg', default='result/Emp_Converge_check/')
    return parser.parse_args()

def set_train_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def train():
    if args.type in ['t1_pos', 't2_pos', 't3_pos', 't1_neg', 't2_neg', 't3_neg']:
        botData_f, N_train, prop_label = load_empirical_data(args.type)
    else:
        raise ValueError("The scenairo is not implemented of the scenario name is incorrect.")

    botData_f = botData_f.to(device)
    model_f = HomoEdgeDetector(in_dim=2, h_dim=args.hm, out_dim=args.dzm).to(device)
    model_g = BotImpact(in_dim=2, h_e=args.he, h_o=args.ho, out_dim=1).to(device)
    model_d = Discriminator(in_dim=args.he, h_dim=args.hd).to(device) # model_d in_dim = model_g h_dim
    optimizer_fg = torch.optim.Adam([{'params': model_f.parameters(), 'lr': args.lr_mf},
                                  {'params': model_g.parameters(), 'lr': args.lr_mg}])
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=args.lr_md)

    # stopping criteria
    ssu = SaveStoppingUtil(args)

    # for counterfactual edge generation
    treat_idx, control_idx = torch.nonzero(botData_f.y[:, 2] == -1).squeeze(), torch.nonzero(botData_f.y[:, 2] == 1).squeeze()

    # split train-val (outcome mask)
    treat_rd_idx, control_rd_idx = torch.randperm(treat_idx.size(0)), torch.randperm(control_idx.size(0))
    Ntreat_train, Ncontrol_train = int(treat_idx.size(0) * args.train_prec), int(control_idx.size(0) * args.train_prec) # sample balancing
    treat_idx_train, control_idx_train = treat_idx[treat_rd_idx[:Ntreat_train]], control_idx[control_rd_idx[:Ncontrol_train]]
    treat_idx_test, control_idx_test = treat_idx[treat_rd_idx[Ntreat_train:]], control_idx[control_rd_idx[Ncontrol_train:]]

    # record training results for model check
    r_y, r_jt, r_cffool, r_jf = 0, 0, 0, 0


    for epoch in range(1, args.max_epoch+1):
        model_f.train()
        model_g.train()
        model_d.train()

        # Counterfactual Graph Generation
        optimizer_fg.zero_grad()
        homo_edge_index, hetero_edge_index, Z_f = model_f(botData_f.x, botData_f.edge_index, mask_homo=args.mask_homo)
        N_homo_edge, N_hetero_edge = homo_edge_index.shape[-1], hetero_edge_index.shape[-1]
        # counterfactual graph generation
        follower, following = random.choices([i for i in range(N_train)], k=N_homo_edge), random.choices([i for i in range(N_train)], k=N_homo_edge)
        cfgraph_edge = generate_counterfactual_edge2(follower, following, hetero_edge_index, device) # for effect estimation
        botData_cf = Data(x=botData_f.x, edge_index=cfgraph_edge.contiguous(), y=botData_f.y).to(device)

        # Outcome Generation
        out_y1, out_yc0, out_y0, out_yc1, Zf, Zcf, treat_prob = model_g(botData_f.x, botData_f.edge_index, botData_cf.x, botData_cf.edge_index, treat_idx_train, control_idx_train) # out_y1, out_yc0: (num_treat_train), out_y0, out_yc1: (num_control_train), *_prob: (num_nodes)

        # Predict the outcome and treatment
        out_y = torch.cat([out_y1, out_y0], dim=-1)
        target_y = torch.cat([botData_f.y[:, 1][treat_idx_train], botData_f.y[:, 1][control_idx_train]])
        loss_y = F.mse_loss(out_y.float(), target_y.float())
        # Fool the discriminator (Balancing Factual & Counterfactual)
        _, fact_prob_cf = model_d(Zf, Zcf)
        loss_cffool = torch.nn.BCELoss()(fact_prob_cf, torch.ones_like(fact_prob_cf))
        # Max the discrepancy between treat/control
        target_t = torch.cat([torch.ones_like(treat_prob[treat_idx_train]),torch.zeros_like(treat_prob[control_idx_train])],dim=0)
        out_t = torch.cat([treat_prob[treat_idx_train], treat_prob[control_idx_train]], dim=0)
        loss_jt = torch.nn.CrossEntropyLoss()(out_t, target_t)

        # Generator Loss
        loss_g = loss_y*args.ly + loss_jt*args.ljt + loss_cffool * args.ljg
        loss_g.backward()
        optimizer_fg.step()

        # Discriminator
        optimizer_d.zero_grad()
        fact_prob, fact_prob_cf = model_d(Zf.detach(), Zcf.detach()) # Prevents gradients from propagating in the generator
        loss_jf = torch.nn.BCELoss()(torch.cat([fact_prob, fact_prob_cf], dim=0), torch.cat([torch.ones_like(fact_prob), torch.zeros_like(fact_prob_cf)], dim=0))
        loss_d = loss_jf * args.ljd
        loss_d.backward()
        optimizer_d.step()
        #print("check loss | ly, ljt, lcff, ljf: {:.4f} {:.4f} {:.4f} {:.4f}".format(loss_y.item(), loss_jt.item(),
        #                                                               loss_cffool.item(), loss_jf.item()))

        r_cffool, r_jf = r_cffool+loss_cffool.item(), r_jf+loss_jf.item()

        # Evaluation
        if epoch%args.eval_per_epoch == 0:
            model_f.eval()
            model_g.eval()
            model_d.eval()

            # model selection on the validation sample
            # out_y1, out_yc0: (num_treat_train), out_y0, out_yc1: (num_control_train), *_prob: (num_nodes)
            out_y1, out_yc0, out_y0, out_yc1, Zf, Zcf, _ = model_g(botData_f.x, botData_f.edge_index, botData_cf.x, botData_cf.edge_index, treat_idx_test, control_idx_test)
            # check the generated sample
            fact_prob, fact_prob_cf = model_d(Zf.detach(), Zcf.detach())
            loss_jf_va = torch.nn.BCELoss()(torch.cat([fact_prob, fact_prob_cf], dim=0), torch.cat([torch.ones_like(fact_prob), torch.zeros_like(fact_prob_cf)], dim=0))

            # estimate results on all samples
            # whether the node have a counterfactual counterpart
            treat_idx_ok, control_idx_ok = match_node(cfgraph_edge, botData_f.y[:, 0], torch.LongTensor(prop_label).to(device), treat_idx, control_idx)
            out_y1, out_yc0, out_y0, out_yc1, Zf, Zcf, _ = model_g(botData_f.x, botData_f.edge_index, botData_cf.x, botData_cf.edge_index, treat_idx_ok, control_idx_ok)

            #print("treat/control: ", treat_idx_ok.shape, control_idx_ok.shape)
            eATE_test, ePEHE_test, treat_eff, ave_treat, ave_control = evaluate_metric(out_y0, out_y1, out_yc1, out_yc0, effect_true=args.effect_true, device=device)
            # model output
            # print("Epoch: " + str(epoch), "Data: ", args.type)
            eATE_test = eATE_test.detach().cpu().numpy()
            ePEHE_test = ePEHE_test.detach().cpu().numpy()
            treat_eff = treat_eff.detach().cpu().numpy()

            # for empirical data (save direct model)
            if args.type in ['t1_pos', 't2_pos', 't3_pos', 't1_neg', 't2_neg', 't3_neg']:
                if args.save_emp_epoch_statedict:
                    ssu.direct_model_save(model_state_dict={
                        'model_f_state_dict': model_f.state_dict(),
                        'model_g_state_dict': model_g.state_dict(),
                        'model_d_state_dict': model_d.state_dict(),
                    }, save_model_path=args.emp_model_save_dir, save_model_name='AdvG_' + args.type + '_e' + str(epoch) + '.ckpt')

                    if args.type in ['t1_neg', 't2_neg', 't3_neg']: # for printing the result
                        treat_eff = -treat_eff # treatment effect: towards polarized direction
                    print('Effect: {:.4f}'.format(treat_eff))
                    print("================================")

                # check: comparison with naive approach
                if args.check_emp_with_Naive:
                    ave_t_naive = torch.mean(botData_f.y[:, 1][treat_idx]).item()
                    ave_c_naive = torch.mean(botData_f.y[:, 1][control_idx]).item()
                    ave_t_adv_wcf = torch.mean(torch.cat([out_y1, out_yc1], dim=0)).item()
                    ave_c_adv_wcf = torch.mean(torch.cat([out_y0, out_yc0], dim=0)).item()
                    ave_t_adv = torch.mean(out_y1).item()
                    ave_c_adv = torch.mean(out_y0).item()
                    if epoch == args.rep_epoch_check:
                        print("Naive        | T: {:.4f}, C: {:.4f}".format(ave_t_naive, ave_c_naive))
                        print("AdvG with CF | T: {:.4f}, C: {:.4f}".format(ave_t_adv_wcf, ave_c_adv_wcf))

            # # check convergence: collect the loss
            # if args.converge_check:
            #     ssu.record_train_msg(epoch, ave_treat, ave_control, treat_eff, eATE_test, ePEHE_test, r_cffool, r_jf, loss_jf_va)
            r_cffool, r_jf = 0, 0

            ssu.update_gold_eval(score_eval = loss_jf_va.detach().cpu().numpy(), epoch_eval = epoch, model_state_dict={
                    'model_f_state_dict': model_f.state_dict(),
                    'model_g_state_dict': model_g.state_dict(),
                    'model_d_state_dict': model_d.state_dict(),
                }, result_dict={
                    'aveT': ave_treat,
                    'aveC': ave_control,
                    'eATE': eATE_test,
                    'ePEHE': ePEHE_test,
                })

            if epoch == args.rep_epoch_check:
                # treatment effect (ATT)
                # print("Num of eval treat: ", len(out_y1), len(out_yc0), len(treat_idx_ok))
                # print("Num of eval control: ", len(out_y0), len(out_yc1), len(control_idx_ok))
                te_treatGroup, te_controlGroup = out_y1 - out_yc0, out_yc1 - out_y0 # TE in treatment group/TE in control group
                # print("ATT | Ave Treat: {:.4f}, Ave Cf Control: {:.4f}".format(torch.mean(out_y1).item(), torch.mean(out_yc0).item()))
                # counterfactual analysis
                if args.type in ['t1_pos', 't2_pos', 't3_pos', 't1_neg', 't2_neg', 't3_neg']:
                    cf_check_path = 'Dataset/twi22/cf_result/'
                    create_dir(cf_check_path)
                    effect_t_dict = pd.DataFrame({'treatNode_tr_id': treat_idx_ok.detach().cpu().numpy(),
                                   'estimatedTE': te_treatGroup.detach().cpu().numpy()})
                    effect_t_dict.to_csv(cf_check_path+'CfA_'+args.type+'_treatGp.csv', index=False)
                    effect_c_dict = pd.DataFrame({'controlNode_tr_id': control_idx_ok.detach().cpu().numpy(),
                                   'estimatedTE': te_controlGroup.detach().cpu().numpy()})
                    effect_c_dict.to_csv(cf_check_path+'CfA_'+args.type+'_controlGp.csv', index=False)
                    #print("Check Consistency: {:.4f} {:.4f}".format(treat_eff, (np.sum(effect_t_dict['estimatedTE']) + np.sum(effect_c_dict['estimatedTE'])) / (len(te_treatGroup) + len(te_controlGroup))))

                # store the results of repeated experiments in simulation
                ssu.get_intermid_eval(model_state_dict={
                    'model_f_state_dict': model_f.state_dict(),
                    'model_g_state_dict': model_g.state_dict(),
                    'model_d_state_dict': model_d.state_dict(),
                }, result_dict={
                    'aveT': ave_treat,
                    'aveC': ave_control,
                    'eATE': eATE_test,
                    'ePEHE': ePEHE_test,
                })


        if epoch == args.max_epoch: # stop training and store results
            if args.converge_check:
                ssu.save_train_msg(save_path=args.conv_check_dir, file_name='AdvG_' + args.type + str(seed) + '.csv')

            if args.type in ['t1_pos', 't2_pos', 't3_pos', 't1_neg', 't2_neg', 't3_neg']:
                ssu.direct_model_save(ssu.Model_dict_gold, save_model_path=args.emp_model_save_dir, save_model_name='AdvG_' + args.type + '_e'+str(ssu.gold_epoch)+'gold.ckpt')


if __name__ == "__main__":
    args = get_train_args()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    seed = 110
    # for empirical data
    print("Prepare training: ")
    set_train_seed(seed)
    train()
