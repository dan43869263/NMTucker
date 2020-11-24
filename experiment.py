import torch
import numpy as np
from pytorchtools import EarlyStopping
import time
from utils import *
from model import *

class Experiment:
    
    def __init__(self, learning_rate=0.0005,shape=(3,4,5,6),rank=(2,3,4,5),validation_split=0.1,
                 num_iterations=500, cuda=True, 
                 model='ML1',batch_size=128,patience=10,core=(2,3,4),ccore=(2,3,4),
                 tr_idxs=1,tr_vals=1,
                 lambda_l1=1e-3,lambda_l2=1e-3,regularization='L1'):
        
        self.learning_rate = learning_rate
        self.shape = shape
        self.rank = rank
        self.core = core
        self.ccore = ccore
        self.num_iterations = num_iterations
        self.cuda = cuda
        self.model = model
        self.batch_size = batch_size
        self.patience=patience
        self.tr_idxs=tr_idxs
        self.tr_vals=tr_vals
        self.validation_split=validation_split
        self.regularization=regularization
        self.lambda_l1=lambda_l1
        self.lambda_l2=lambda_l2
             
    def save_models(epoch):
        torch.save(model.state_dict(), "./adclickmodel_{}.model".format(epoch))
        print("Chekcpoint saved")
        
    def get_index(self, data, idx):
        batch = data[idx:idx+self.batch_size,:]
        targets = data[idx:idx+self.batch_size,-1]
        targets = torch.FloatTensor(targets)
        if self.cuda:
           targets = targets.cuda()
        return np.array(batch), targets

    def evaluate(self, model, idxs,vals):
        with torch.no_grad():
            model.eval()
            #prob_list=[]
            prediction_list=[]
            target_list=[]
            losses=[]
            rmse_list=[]
            mae_list=[]
            
            data=np.concatenate((idxs,vals.reshape(-1,1)),axis=1)
            test_data_idxs = data
            #print("Number of data points: %d" % len(test_data_idxs))
        
            for i in range(0, len(test_data_idxs),self.batch_size):
                data_batch, targets = self.get_index(test_data_idxs, i)
                u_idx = torch.tensor(data_batch[:,0].astype(int))
                v_idx = torch.tensor(data_batch[:,1].astype(int))
                w_idx = torch.tensor(data_batch[:,2].astype(int))
                
                if self.cuda:
                    u_idx = u_idx.cuda()
                    v_idx = v_idx.cuda()
                    w_idx = w_idx.cuda()
                    
                predictions = model.forward(u_idx, v_idx,w_idx) 
            
                prediction_list=prediction_list+predictions.cpu().tolist() 
                target_list=target_list+targets.cpu().tolist()

                loss = model.loss(predictions, targets)
                losses.append(loss.item())
                        
            val_loss=np.mean(losses)  
            val_mae=mae(target_list, prediction_list)
            val_rmse=rmse(target_list, prediction_list)
            val_mape=mape(target_list, prediction_list)

            return prediction_list,val_loss,val_rmse,val_mae,val_mape 
    
    def train_and_eval(self):
        train_losses=[]
        val_losses=[]
        val_rmses=[]
        val_maes=[]
        val_mapes=[]
        
        print("Training the NMTucker model...")
        print("batch size is {0}".format(self.batch_size))
        train,val=train_val_split(self.tr_idxs,self.tr_vals,self.validation_split)
        train_data_idxs = train
        print("Number of training data points: %d" % len(train_data_idxs))
        
        if self.model=='ML1': #NMTucker-L1 
            model = ML1(self.shape, self.rank)
            
        elif self.model=="ML2": #NMTucker-L2
            model = ML2(self.shape, self.rank,self.core)
            
        elif self.model=="ML3": #NMTucker-L3
            model = ML3(self.shape, self.rank,self.core,self.ccore)
             
        if self.cuda:
            model.cuda()
            
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        print("Starting training...")
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        for it in range(1, self.num_iterations+1):
            start_train = time.time()
            model.train()    
            losses = []
            np.random.shuffle(train_data_idxs)
            
            for j in range(0, len(train_data_idxs),self.batch_size):
                data_batch, targets = self.get_index(train_data_idxs,j)
                opt.zero_grad()
                u_idx = torch.tensor(data_batch[:,0].astype(int))
                v_idx = torch.tensor(data_batch[:,1].astype(int))
                w_idx = torch.tensor(data_batch[:,2].astype(int))
                
                if self.cuda:
                    u_idx = u_idx.cuda()
                    v_idx = v_idx.cuda()
                    w_idx = w_idx.cuda()
                    
                predictions = model.forward(u_idx, v_idx , w_idx)  
                loss = model.loss(predictions, targets)
                losses.append(loss.item())
                
                if self.regularization=='L1':
                    lossl1=l1_regularizer(model, self.lambda_l1)
                    loss=loss+lossl1
                    
                elif self.regularization=='L2':
                    lossl2=l2_regularizer(model, self.lambda_l2)
                    loss=loss+lossl2
                    
                loss.backward(retain_graph=True)
                opt.step()
                
            train_loss=np.mean(losses)
            train_losses.append(train_loss)
            
            print('\nITERATION:', it)
            print('TIME ELAPSED:{:.4f}'.format(time.time()-start_train))    
            print('TRAINING LOSS:{:.7f}'.format(train_loss))
            
            with torch.no_grad():
                
                _,val_loss,val_rmse,val_mae,val_mape = self.evaluate(model,val[:,0:3],val[:,-1])
                print('VALIDATION LOSS:{:.7f}'.format(val_loss))
                val_losses.append(val_loss)
                val_rmses.append(val_rmse)
                val_maes.append(val_mae)
                val_mapes.append(val_mape)
                
                print('val_loss: {:.7f}'.format(val_loss))
                print('val_rmse: {:.7f}'.format(val_rmse))
                print('val_mae: {:.7f}'.format(val_mae))
                print('val_mape: {:.7f}'.format(val_mape))

                # early_stopping needs the validation loss to check if it has decresed, 
                # and if it has, it will make a checkpoint of the current model
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                   print("Early stopping")
                   break
                    
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load('checkpoint.pt'))
                    
        dic = dict()
        dic['train_loss'] = train_losses
        dic['val_loss'] = val_losses
        dic['val_rmse'] = val_rmses
        dic['val_mae'] = val_maes
        dic['val_mape'] = val_mapes
        dic['model']=model
        return dic