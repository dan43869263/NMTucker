import os
import json
import zipfile
from pprint import pprint
import numpy as np
import tensorflow.compat.v1 as tf
import keras as k

def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100


def mape_keras(y_true, y_pred, threshold=0.1):
    v = k.backend.clip(k.backend.abs(y_true), threshold, None)
    diff = k.backend.abs((y_true - y_pred) / v)
    return 100.0 * k.backend.mean(diff, axis=-1)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

def mape(y_true, y_pred, threshold=0.1):
    v = np.clip(np.abs(y_true), threshold, None)
    diff = np.abs((y_true - y_pred) / v)
    return 100.0 * np.mean(diff, axis=-1).mean()

def transform(idxs):
    return [idxs[:, i] for i in range(idxs.shape[1])]

def set_session(device_count=None, seed=0):
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(
            gpu_options=gpu_options, 
            device_count=device_count)

def get_metrics(model, x, y, batch_size=1024):
    yp = model.predict(x, batch_size=batch_size, verbose=1).flatten()
    return {
        "rmse": float(rmse(y, yp)), 
        "mape": float(mape(y, yp)), 
        "mae": float(mae(y, yp))
    }, rmse(y, yp), mape(y, yp), mae(y, yp)

def create_costco(shape, rank, nc):
    inputs = [k.Input(shape=(1,), dtype="int32") for i in range(len(shape))]
    embeds = [
        k.layers.Embedding(output_dim=rank, input_dim=shape[i])(inputs[i])
        for i in range(len(shape))
    ]
    x = k.layers.Concatenate(axis=1)(embeds)
    x = k.layers.Reshape(target_shape=(rank, len(shape), 1))(x)
    x = k.layers.Conv2D(
        nc, 
        kernel_size=(1, len(shape)), 
        activation="relu", 
        padding="valid"
    )(x)
    x = k.layers.Conv2D(
        nc, 
        kernel_size=(rank, 1), 
        activation="relu", 
        padding="valid"
    )(x)
    x = k.layers.Flatten()(x)
    x = k.layers.Dense(nc, activation="relu")(x)
    outputs = k.layers.Dense(1, activation="relu")(x)
    model = k.Model(inputs=inputs, outputs=outputs)

    return model



def print_results(train_rmse_list, train_mae_list, train_mape_list,
                          test_rmse_list, test_mae_list, test_mape_list):
    
    print("PRINTING FINAL RESULTS...\n")
    
    print('training rmse', train_rmse_list, '\n')
    print('mean training rmse and std. dev:', np.mean(train_rmse_list), np.std(train_rmse_list, ddof = 1), '\n')
          
    print('training mae', train_mae_list, '\n')
    print('mean training mae and std. dev:', np.mean(train_mae_list), np.std(train_mae_list, ddof = 1), '\n')
          
    print('training mape', train_mape_list, '\n')
    print('mean training rmse and std. dev:',np.mean(train_mape_list), np.std(train_mape_list, ddof = 1), '\n')
    
    print('test rmse', test_rmse_list, '\n')
    print('mean test rmse and std. dev:', np.mean(test_rmse_list), np.std(test_rmse_list, ddof = 1), '\n')
    
    print('test mae', test_mae_list, '\n')
    print('mean test mae and std. dev:', np.mean(test_mae_list), np.std(test_mae_list, ddof = 1), '\n')
          
    print('test mape', test_mape_list, '\n')
    print('mean test rmse and std. dev:', np.mean(test_mape_list), np.std(test_mape_list, ddof = 1), '\n')