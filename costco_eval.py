from costco_utils import *
from utils import *
import tensorflow.compat.v1 as tf
import keras as k
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--rank", type=int, default= 20, help= "rank")
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

for i in range(2): # Do the experiment 10 times
    
    lr = 1e-4
    epochs = 50
    batch_size = 256
    seed = 3
    verbose = 1
    rank = args.rank
    nc = rank
    set_session(device_count={"GPU": 0}, seed=seed)
    optim = k.optimizers.Adam(lr=lr)

    model = create_costco(shape, rank, nc)
    model.compile(optim, loss=["mse"], metrics=["mae", mape_keras])
    hists = model.fit(
    x=transform(tr_idxs),
    y=tr_vals,
    verbose=verbose,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,
    callbacks=[k.callbacks.EarlyStopping(
        monitor="val_mean_absolute_error", 
        patience=10, 
        restore_best_weights=True)],
        );

    tr_info, rmse, mape, mae = get_metrics(model, transform(tr_idxs), tr_vals)
    
    train_rmse_list.append(rmse)
    train_mae_list.append(mae)
    train_mape_list.append(mape)
    
    te_info, rmse, mape, mae = get_metrics(model, transform(te_idxs), te_vals)
    
    test_rmse_list.append(rmse)
    test_mae_list.append(mae)
    test_mape_list.append(mape)
    
    pprint({'train': tr_info, 'test': te_info})
    
print_results(train_rmse_list, train_mae_list, train_mape_list,
                      test_rmse_list, test_mae_list, test_mape_list)
    
