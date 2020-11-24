import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_loss(dic):
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(dic['train_loss'], label='Training loss ')
    plt.plot(dic['val_loss'], label='Validation loss ')
    plt.legend(frameon=False)
    plt.title('train and validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(dic['val_mae'])
    plt.title('validation MAE')
    plt.xlabel('epoch')
    plt.ylabel('MAE')
    plt.show()

def unfold(tensor,mode):
    return torch.reshape(tensor.transpose(mode, 0), (tensor.shape[mode], -1))

def vec_to_tensor(vec, shape):
    """Folds a vectorised tensor back into a tensor of shape `shape`
    
    Parameters
    ----------
    vec : 1D-array
        vectorised tensor of shape ``(i_1 * i_2 * ... * i_n)``
    shape : tuple
        shape of the ful tensor
    
    Returns
    -------
    ndarray
        tensor of shape `shape` = ``(i_1, ..., i_n)``
    """
    return torch.reshape(vec, shape)

def fold(unfolded_tensor, mode, shape):
    """Refolds the mode-`mode` unfolding into a tensor of shape `shape`
    
        In other words, refolds the n-mode unfolded tensor
        into the original tensor of the specified shape.
    
    Parameters
    ----------
    unfolded_tensor : ndarray
        unfolded tensor of shape ``(shape[mode], -1)``
    mode : int
        the mode of the unfolding
    shape : tuple
        shape of the original tensor before unfolding
    
    Returns
    -------
    ndarray
        folded_tensor of shape `shape`
    """
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    t= torch.reshape(unfolded_tensor, full_shape)
    return t.transpose(0,mode)

def mode_dot(tensor, matrix_or_vector, mode):
        """n-mode product of a tensor and a matrix or vector at the specified mode

        Mathematically: :math:`\\text{tensor} \\times_{\\text{mode}} \\text{matrix or vector}`


        Parameters
        ----------
        tensor : ndarray
            tensor of shape ``(i_1, ..., i_k, ..., i_N)``
        matrix_or_vector : ndarray
            1D or 2D array of shape ``(J, i_k)`` or ``(i_k, )``
            matrix or vectors to which to n-mode multiply the tensor
        mode : int

        Returns
        -------
        ndarray
            `mode`-mode product of `tensor` by `matrix_or_vector`
            * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)` if matrix_or_vector is a matrix
            * of shape :math:`(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)` if matrix_or_vector is a vector

        See also
        --------
        multi_mode_dot : chaining several mode_dot in one call
        """
        # the mode along which to fold might decrease if we take product with a vector
        fold_mode = mode
        new_shape = list(tensor.shape)

        if len(matrix_or_vector.shape) == 2:  # Tensor times matrix
            # Test for the validity of the operation
            if matrix_or_vector.shape[1] != tensor.shape[mode]:
                raise ValueError(
                    'shapes {0} and {1} not aligned in mode-{2} multiplication: {3} (mode {2}) != {4} (dim 1 of matrix)'.format(
                        tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[1]
                    ))
            new_shape[mode] = matrix_or_vector.shape[0]
            vec = False

        elif len(matrix_or_vector.shape) == 1:  # Tensor times vector
            if matrix_or_vector.shape[0] != tensor.shape[mode]:
                raise ValueError(
                    'shapes {0} and {1} not aligned for mode-{2} multiplication: {3} (mode {2}) != {4} (vector size)'.format(
                        tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[0]
                    ))
            if len(new_shape) > 1:
                new_shape.pop(mode)
            else:
                # Ideally this should be (), i.e. order-0 tensors
                # MXNet currently doesn't support this though..
                new_shape = []
            vec = True

        else:
            raise ValueError('Can only take n_mode_product with a vector or a matrix.'
                             'Provided array of dimension {} not in [1, 2].'.format(T.ndim(matrix_or_vector)))

        res = torch.mm(matrix_or_vector, unfold(tensor, mode))

        if vec: # We contracted with a vector, leading to a vector
            return vec_to_tensor(res, shape=new_shape)
        else: # tensor times vec: refold the unfolding
            return fold(res, fold_mode, new_shape)
        

def multi_mode_dot(tensor, matrix_or_vec_list, modes=None, skip=None):
    if modes is None:
        modes = range(len(matrix_or_vec_list))

    decrement = 0  # If we multiply by a vector, we diminish the dimension of the tensor

    res = tensor

    # Order of mode dots doesn't matter for different modes
    # Sorting by mode shouldn't change order for equal modes
    factors_modes = sorted(zip(matrix_or_vec_list, modes), key=lambda x: x[1])
    for i, (matrix_or_vec, mode) in enumerate(factors_modes):
        if (skip is not None) and (i == skip):
            continue

        if mode==2:
            res = mode_dot(res, matrix_or_vec, mode - decrement)
        else:
            res =mode_dot(res, matrix_or_vec, mode - decrement)
            #res=torch.nn.functional.leaky_relu_(torch.tensor(res)) 
            #res=res.detach().numpy()

        if np.ndim(matrix_or_vec) == 1:
            decrement = decrement+ 1

    return res

def mape(y_true, y_pred, threshold=0.1):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    v = np.clip(np.abs(y_true), threshold, None)
    diff = np.abs((y_true - y_pred) / v)
    return  100.0 * np.mean(diff, axis=-1).mean()

def mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_pred - y_true))

def rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

#use relu or leaky_relu function
def nonlinear(active,vector):
    if active=='relu':
        return torch.nn.functional.relu_(vector)
    elif active == 'leaky_relu':
        return torch.nn.functional.leaky_relu_(vector)
    
#use relu or leaky_relu layer    
def non_layer(active):
    if active=='relu':
        return nn.ReLU()
    elif active=='leaky_relu':
        return nn.LeakyReLU()
    
def train_val_split(tr_idxs,tr_vals,validation_split):
    validation_split = validation_split
    dataset_size = len(tr_vals)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(1337)
    np.random.shuffle(indices)
    train_indices, valid_indices = indices[split:], indices[:split]
    tr_idx=tr_idxs[train_indices]
    tr_val=tr_vals[train_indices].reshape(-1,1)
    val_idx=tr_idxs[valid_indices]
    val_val=tr_vals[valid_indices].reshape(-1,1)
    trainset=np.concatenate((tr_idx,tr_val),axis=1)
    valset=np.concatenate((val_idx,val_val),axis=1)
    return trainset,valset

def get_result(experiment,model,tr_idxs,tr_vals):
    _,mse,rmse,mae,mape=experiment.evaluate(model,tr_idxs,tr_vals)
    print('mse:',mse,'rmse:',rmse,'mae:',mae,'mape:',mape)
    return mse,rmse,mae,mape
    
def l1_regularizer(model, lambda_l1):
    lossl1 = 0
    for model_param_name, model_param_value in model.named_parameters():
            #if model_param_name.endswith('weight'):
            lossl1 += lambda_l1 * model_param_value.abs().sum()
    return lossl1    

def l2_regularizer(model, lambda_l2):
    lossl2 = 0
    for model_param_name, model_param_value in model.named_parameters():
            #if model_param_name.endswith('weight'):
            lossl2 += lambda_l2 * (model_param_value**2).sum()
    return lossl2


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