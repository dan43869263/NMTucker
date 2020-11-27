from model import *
from utils import *
from experiment import Experiment

import torch
import numpy as np
import pandas as pd
from pytorchtools import EarlyStopping
import time
import argparse
import matplotlib.pyplot as plt
torch.cuda.is_available() #check cuda

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--num_iterations", type=int, default=200, nargs="?",
                help="Number of iterations.")
parser.add_argument("--patience", type=int, default=5, nargs="?",
                help="patience for early stop.")
parser.add_argument("--batch_size", type=int, default=256, nargs="?",
                help="Batch size.")
parser.add_argument("--lr", type=float, default=1e-4, nargs="?",
                help="Learning rate.")
parser.add_argument("--rank", default=(5,5,5), nargs="?",
                help="For NMTucker-L1, NMTUcker-L2, NMtucker-L3, rank is the shape of core tensor G in the output layer.")
parser.add_argument("--core", default=(3,3,3), nargs="?",
                help="For NMTucker-L2 core is the shape of core tensor G1 in the first layer. For NMTucker-L3, core is the shape of core tensor G2 in the second layer")
parser.add_argument("--ccore", default=(4,4,4), nargs="?",
                help="For NMTucker-L3, ccore is the shape of the core tensor G1 in the first layer")
parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                help="Whether to use cuda (GPU) or not (CPU).")  
parser.add_argument("--validation_split", type=float, default=0.1, nargs="?",
                help="validation split ratio")
parser.add_argument("--model", type=str, default='ML1', nargs="?",
                help="use which model:ML1,ML2,ML3")
parser.add_argument("--dataset", type=str, default='sg', nargs="?",
                help="'sg','netflix' or 'ccds'")
args = parser.parse_args()


tr_idxs, tr_vals, te_idxs, te_vals, shape = load_dataset(args.dataset)
   

train_rmse_list = []
train_mae_list = []
train_mape_list = []
test_rmse_list = []
test_mae_list = []
test_mape_list = []

for i in range(10): # do the experiment 10 times
    
    print('EXPERIMENT', (i + 1))
    ex = Experiment(num_iterations = args.num_iterations,
                    batch_size = args.batch_size,
                    learning_rate = args.lr, 
                    shape = shape,
                    rank = args.rank,
                    core = args.core,
                    ccore = args.ccore,
                    cuda = args.cuda,
                    model = args.model,
                    patience = args.patience,
                    tr_idxs = tr_idxs,
                    tr_vals = tr_vals,
                    validation_split  =args.validation_split,
                    regularization=None)

    dic = ex.train_and_eval()
    
    print('TRAINING RESULTS:')
    mse,rmse,mae,mape = get_result(ex, dic['model'], tr_idxs, tr_vals) #print and store training results  
    
    train_rmse_list.append(rmse)
    train_mae_list.append(mae)
    train_mape_list.append(mape)
    
    print('TEST RESULTS:') 
    mse,rmse,mae,mape = get_result(ex,dic['model'],te_idxs,te_vals) #print and store training results
    
    test_rmse_list.append(rmse)
    test_mae_list.append(mae)
    test_mape_list.append(mape)
    
print_results(train_rmse_list, train_mae_list, train_mape_list,
              test_rmse_list, test_mae_list, test_mape_list) # print_results function is from utils.py
