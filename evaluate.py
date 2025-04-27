import torch
import numpy as np
import pandas as pd


def evaluate_metric(pred_0, pred_1, pred_c1, pred_c0, effect_true, device):
    tau_pred = torch.cat([pred_c1, pred_1], dim=0) - torch.cat([pred_0, pred_c0], dim=0)

    ave_treat = torch.mean(torch.cat([pred_1, pred_c1], dim=0)).item()
    ave_control = torch.mean(torch.cat([pred_0, pred_c0], dim=0)).item()

    #print("treat ave: {:.4f}".format(ave_treat))
    #print("control ave: {:.4f}".format(ave_control))
    #print("--------------------------------")
    tau_true = torch.ones(tau_pred.shape).to(device) * effect_true
    ePEHE = torch.sqrt(torch.mean(torch.square(tau_pred-tau_true)))
    eATE = torch.abs(torch.mean(tau_pred) - torch.mean(tau_true))
    Treat_eff = torch.mean(tau_pred)
    return eATE, ePEHE, Treat_eff, ave_treat, ave_control


class RepeatResultCollection():
    def __init__(self):
        # metrics to be check or evaluated
        self.Data_id = []
        self.eATE, self.ePEHE = [], []
        self.aveT, self.aveC = [], []

    def store_add(self, DATA_ID, result_dict):
        self.Data_id.append(DATA_ID)
        self.aveT.append(result_dict['aveT'])
        self.aveC.append(result_dict['aveC'])
        self.eATE.append(result_dict['eATE'])
        self.ePEHE.append(result_dict['ePEHE'])

    def store_repeat_resdt(self, args, print_res):
        repeated_result = {
            'Data_id': self.Data_id,
            'aveT': self.aveT,
            'aveC': self.aveC,
            'eATE': self.eATE,
            'ePEHE': self.ePEHE
        }
        if print_res:
            print(" ---------------------------------------------------------------------- ")
            print("Data: ", args.type)
            print(" eATE: {:.4f} ({:.4f})".format(np.mean(self.eATE), np.std(self.eATE)),
                  " ePEHE: {:.4f} ({:.4f})".format(np.mean(self.ePEHE), np.std(self.ePEHE)))
        repeated_result = pd.DataFrame(repeated_result)
        return repeated_result