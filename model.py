import torch
from torch.nn.init import xavier_normal_
import numpy as np
from utils import *

#NMTucker-L1  our 1 layer NMTucker model
class ML1(torch.nn.Module):
    def __init__(self,shape,rank):
        super(ML1, self).__init__()
        #Embedding module-initial three embedding matrix U,V,W
        self.U = torch.nn.Embedding(shape[0], rank[0], padding_idx=0) #the mode 1 factor matrix U
        self.V = torch.nn.Embedding(shape[1], rank[1], padding_idx=0) #the mode 2 factor matrix V
        self.W = torch.nn.Embedding(shape[2], rank[2], padding_idx=0) #the mode 3 factor matrix W
        self.G = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (rank[0], rank[1], rank[2])), 
                                     dtype=torch.float,
                                     device="cuda", requires_grad=True)) #the core tensor G

        self.loss = torch.nn.MSELoss() #we use the mse loss   
        self.d1=rank[0]
        self.d2=rank[1]
        self.d3=rank[2]
        
    def init(self):
        xavier_normal_(self.U.weight.data)
        xavier_normal_(self.V.weight.data)
        xavier_normal_(self.W.weight.data)
   
    def forward(self, u_idx, v_idx, w_idx):
        #Embedding model -extract three embed-ding vectors (for here is three batches of vectors)
        u = self.U(u_idx) 
        u = u.view(-1,self.d1) #b*d1
        v = self.V(v_idx)
        v = v.view(-1,1,self.d2) #b*1*d2
        w = self.W(w_idx)
        w = w.view(-1,1,self.d3) #b*1*d3
        
      
        #Nonlinear Tucker multiplication module
        x = torch.mm(u, self.G.view(self.d1,-1)) #b*d1 mm d1*(d2*d3) = b*(d2*d3)
        x = torch.sigmoid(x) #we use the sigmoid function as activation function, you can also try others like ReLU, Leaky ReLU
        x = x.view(-1,self.d2,self.d3) # b*d2*d3

        G_mat = torch.bmm(v,x) #b*1*d2 bmm b*d2*d3 = b*1*d3
        G_mat = torch.sigmoid(G_mat)
        G_mat = G_mat.view(-1,self.d3,1) #b*d3*1

        x = torch.bmm(w, G_mat) #b*1*d3 bmm b*d3*1=b*1*1
        pred = torch.squeeze(x)
        return pred
        
#NMTucker-L2  our 2 layer NMTucker model
class ML2(torch.nn.Module):
    def __init__(self,shape,rank,core):
        super(ML2, self).__init__()
        #Embedding module-initial three embedding matrix U1,V1,W1 in the first layer and the core tensor G1 in the first layer.
        self.U1 = torch.nn.Embedding(rank[0], core[0], padding_idx=0).cuda() 
        self.V1 = torch.nn.Embedding(rank[1], core[1], padding_idx=0).cuda()
        self.W1 = torch.nn.Embedding(rank[2], core[2], padding_idx=0).cuda()
        self.G1 = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (core[0], core[1], core[2])), 
                                     dtype=torch.float,
                                     device="cuda", requires_grad=True))
        
        #Embedding moduke-initial three embedding matrix U,V,W in the second layer (output layer).
        self.U = torch.nn.Embedding(shape[0], rank[0], padding_idx=0)
        self.V = torch.nn.Embedding(shape[1], rank[1], padding_idx=0)
        self.W = torch.nn.Embedding(shape[2], rank[2], padding_idx=0)

        self.loss = torch.nn.MSELoss()
        self.d1=rank[0]
        self.d2=rank[1]
        self.d3=rank[2]
        self.rank=rank
        self.core=core

    def init(self):
        xavier_normal_(self.U1.weight.data)
        xavier_normal_(self.V1.weight.data)
        xavier_normal_(self.W1.weight.data)
        xavier_normal_(self.U.weight.data)
        xavier_normal_(self.V.weight.data)
        xavier_normal_(self.W.weight.data)
   

 
    def forward(self, u_idx, v_idx, w_idx):
        #Embedding model -extract three embedding vectors (for here is three batches of vectors)
        u = self.U(u_idx) 
        u = u.view(-1,self.d1) #b*d1
        v = self.V(v_idx)
        v = v.view(-1,1,self.d2) #b*1*d2
        w = self.W(w_idx)
        w = w.view(-1,1,self.d3) #b*1*
        
        #reconstruct the core tensor G in the output layer (for more information see the section "Implicit Regularization with Multiple LayerRecursive NMTucker")
        res = mode_dot(self.G1,self.U1.weight,0)
        res = torch.sigmoid(res)
        res = mode_dot(res,self.V1.weight,1)
        res = torch.sigmoid(res)
        G = mode_dot(res,self.W1.weight,2) #the reconstructed core tensor G in the output layer
        G = torch.sigmoid(G)
        
        #Nonlinear Tucker multiplication module
        x = torch.mm(u, G.reshape(self.d1,-1)) #b*d1 mm d1*(d2*d3) = b*(d2*d3)
        x = torch.sigmoid(x) #the first non-linear activation function, for others like ReLU use torch.nn.functional.relu_()
        x = x.view(-1,self.d2,self.d3) # b*d2*d3
   
        G_mat = torch.bmm(v,x) #b*1*d2 bmm b*d2*d3 = b*1*d3
        G_mat = torch.sigmoid(G_mat)# the second non-linear activation function
        G_mat = G_mat.view(-1,self.d3,1) #b*d3*1

        x = torch.bmm(w, G_mat) #b*1*d3 bmm b*d3*1=b*1*1
        pred = torch.squeeze(x)
        return pred
    
#NMTucker-L3  our 3 layer NMTucker model
class ML3(torch.nn.Module):
    def __init__(self,shape,rank,core,ccore):
        super(ML3, self).__init__()
        
        #Embedding module-initial three embedding matrix U1,V1,W1 in the first layer and the core tensor G1 in the first layer.
        self.U1 = torch.nn.Embedding(core[0], ccore[0],padding_idx=0).cuda()
        self.V1 = torch.nn.Embedding(core[1], ccore[1], padding_idx=0).cuda()
        self.W1 = torch.nn.Embedding(core[2], ccore[2], padding_idx=0).cuda()
        self.G1 = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (ccore[0], ccore[1], ccore[2])), 
                                     dtype=torch.float,
                                     device="cuda", requires_grad=True))
        
        #Embedding module-initial three embedding matrix U2,V2,W2 in the second layer.
        self.U2 = torch.nn.Embedding(rank[0], core[0], padding_idx=0).cuda()
        self.V2 = torch.nn.Embedding(rank[1], core[1], padding_idx=0).cuda()
        self.W2 = torch.nn.Embedding(rank[2], core[2], padding_idx=0).cuda()
        
        #Embedding moduke-initial three embedding matrix U,V,W in the third layer (output layer).
        self.U = torch.nn.Embedding(shape[0], rank[0], padding_idx=0)
        self.V = torch.nn.Embedding(shape[1], rank[1], padding_idx=0)
        self.W = torch.nn.Embedding(shape[2], rank[2], padding_idx=0)


        self.loss = torch.nn.MSELoss()
        self.d1=rank[0]
        self.d2=rank[1]
        self.d3=rank[2]
        self.rank=rank
        self.core=core
       
    def init(self):
        xavier_normal_(self.U1.weight.data)
        xavier_normal_(self.V1.weight.data)
        xavier_normal_(self.W1.weight.data)
        xavier_normal_(self.U2.weight.data)
        xavier_normal_(self.V2.weight.data)
        xavier_normal_(self.W2.weight.data)
        xavier_normal_(self.U.weight.data)
        xavier_normal_(self.V.weight.data)
        xavier_normal_(self.W.weight.data)
   
    def forward(self, u_idx, v_idx, w_idx):
        #Embedding model -extract three embedding vectors (for here is three batches of vectors)
        u = self.U(u_idx) 
        u = u.view(-1,self.d1) #b*d1
        v = self.V(v_idx)
        v = v.view(-1,1,self.d2) #b*1*d2
        w = self.W(w_idx)
        w = w.view(-1,1,self.d3) #b*1*d3
           
#reconstruct the core tensor G2 in the second layer (for more information see the section "Implicit Regularization with Multiple LayerRecursive NMTucker")
        G2=torch.tensor(multi_mode_dot(self.G1,
                                       [self.U1(torch.tensor(range(self.core[0])).cuda()),
                                        self.V1(torch.tensor(range(self.core[1])).cuda()),
                                        self.W1(torch.tensor(range(self.core[2])).cuda())],
                                       modes=[0,1,2]))
    
        #reconstruct the core tensor G in the output layer
        G=torch.tensor(multi_mode_dot(G2,
                                     [self.U2(torch.tensor(range(self.rank[0])).cuda()),
                                      self.V2(torch.tensor(range(self.rank[1])).cuda()),
                                      self.W2(torch.tensor(range(self.rank[2])).cuda())],
                                      modes=[0,1,2]))
                                  
        #Nonlinear Tucker multiplication module
        x = torch.mm(u, G.contiguous().view(self.d1,-1)) #b*d1 mm d1*(d2*d3) = b*(d2*d3)
        x = torch.nn.functional.sigmoid(x) #first non-linear activation function
        x = x.view(-1,self.d2,self.d3) # b*d2*d3

        G_mat = torch.bmm(v,x) #b*1*d2 bmm b*d2*d3 = b*1*d3
        G_mat = torch.nn.functional.sigmoid(G_mat) #second non-linear activation function
        G_mat = G_mat.view(-1,self.d3,1) #b*d3*1

        x = torch.bmm(w, G_mat) #b*1*d3 bmm b*d3*1=b*1*1
        pred = torch.squeeze(x)
        return pred

    
#NMTucker-L2 with no activation function except for the output layer.

class ML2_linear(torch.nn.Module):
    def __init__(self,shape,rank,core):
        super(ML2, self).__init__()
        #Embedding module-initial three embedding matrix U1,V1,W1 in the first layer and the core tensor G1 in the first layer.
        self.U1 = torch.nn.Embedding(rank[0], core[0], padding_idx=0).cuda() 
        self.V1 = torch.nn.Embedding(rank[1], core[1], padding_idx=0).cuda()
        self.W1 = torch.nn.Embedding(rank[2], core[2], padding_idx=0).cuda()
        self.G1 = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (core[0], core[1], core[2])), 
                                     dtype=torch.float,
                                     device="cuda", requires_grad=True))
        
        #Embedding moduke-initial three embedding matrix U,V,W in the second layer (output layer).
        self.U = torch.nn.Embedding(shape[0], rank[0], padding_idx=0)
        self.V = torch.nn.Embedding(shape[1], rank[1], padding_idx=0)
        self.W = torch.nn.Embedding(shape[2], rank[2], padding_idx=0)

        self.loss = torch.nn.MSELoss()
        self.d1=rank[0]
        self.d2=rank[1]
        self.d3=rank[2]
        self.rank=rank
        self.core=core

    def init(self):
        xavier_normal_(self.U1.weight.data)
        xavier_normal_(self.V1.weight.data)
        xavier_normal_(self.W1.weight.data)
        xavier_normal_(self.U.weight.data)
        xavier_normal_(self.V.weight.data)
        xavier_normal_(self.W.weight.data)
   

    def forward(self, u_idx, v_idx, w_idx):
        #Embedding model -extract three embedding vectors (for here is three batches of vectors)
        u = self.U(u_idx) 
        u = u.view(-1,self.d1) #b*d1
        v = self.V(v_idx)
        v = v.view(-1,1,self.d2) #b*1*d2
        w = self.W(w_idx)
        w = w.view(-1,1,self.d3) #b*1*
        
        #reconstruct the core tensor G in the output layer (for more information see the section "Implicit Regularization with Multiple LayerRecursive NMTucker")
        res=mode_dot(self.G1,self.U1.weight,0)
        res=mode_dot(res,self.V1.weight,1)
        G=mode_dot(res,self.W1.weight,2) #the reconstructed core tensor G in the output layer
        
        #Nonlinear Tucker multiplication module
        x = torch.mm(u, G.reshape(self.d1,-1)) #b*d1 mm d1*(d2*d3) = b*(d2*d3)
        x = torch.nn.functional.sigmoid(x) #the first non-linear activation function, for others like ReLU use torch.nn.functional.relu_()
        x = x.view(-1,self.d2,self.d3) # b*d2*d3
   
        G_mat = torch.bmm(v,x) #b*1*d2 bmm b*d2*d3 = b*1*d3
        G_mat = torch.nn.functional.sigmoid(G_mat)# the second non-linear activation function
        G_mat = G_mat.view(-1,self.d3,1) #b*d3*1

        x = torch.bmm(w, G_mat) #b*1*d3 bmm b*d3*1=b*1*1
        pred = torch.squeeze(x)
        return pred
    
    
class ML2_random_permute(torch.nn.Module): #this one is not complete yet
    def __init__(self,shape,rank,core):
        super(ML2_random_permute, self).__init__()
        #Embedding module-initial three embedding matrix U1,V1,W1 in the first layer and the core tensor G1 in the first layer.
        self.U1 = torch.nn.Embedding(rank[0], core[0], padding_idx=0).cuda() 
        self.V1 = torch.nn.Embedding(rank[1], core[1], padding_idx=0).cuda()
        self.W1 = torch.nn.Embedding(rank[2], core[2], padding_idx=0).cuda()
        self.G1 = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (core[0], core[1], core[2])), 
                                     dtype=torch.float,
                                     device="cuda", requires_grad=True))
        
        #Embedding moduke-initial three embedding matrix U,V,W in the second layer (output layer).
        self.U = torch.nn.Embedding(shape[0], rank[0], padding_idx=0)
        self.V = torch.nn.Embedding(shape[1], rank[1], padding_idx=0)
        self.W = torch.nn.Embedding(shape[2], rank[2], padding_idx=0)

        self.loss = torch.nn.MSELoss()
        self.d1=rank[0]
        self.d2=rank[1]
        self.d3=rank[2]
        self.rank=rank
        self.core=core

    def init(self):
        xavier_normal_(self.U1.weight.data)
        xavier_normal_(self.V1.weight.data)
        xavier_normal_(self.W1.weight.data)
        xavier_normal_(self.U.weight.data)
        xavier_normal_(self.V.weight.data)
        xavier_normal_(self.W.weight.data)
   

    def forward(self, u_idx, v_idx, w_idx):
        #Embedding model -extract three embedding vectors (for here is three batches of vectors)
        u = self.U(u_idx) 
        u = u.view(-1,self.d1) #b*d1
        v = self.V(v_idx)
        v = v.view(-1,1,self.d2) #b*1*d2
        w = self.W(w_idx)
        w = w.view(-1,1,self.d3) #b*1*
        
        #reconstruct the core tensor G in the output layer (for more information see the section "Implicit Regularization with Multiple LayerRecursive NMTucker")
        res=mode_dot(self.G1,self.U1.weight,0)
        res=mode_dot(res,self.V1.weight,1)
        G=mode_dot(res,self.W1.weight,2) #the reconstructed core tensor G in the output layer
        
        #Nonlinear Tucker multiplication module
        x = torch.mm(u, G.reshape(self.d1,-1)) #b*d1 mm d1*(d2*d3) = b*(d2*d3)
        x = torch.nn.functional.sigmoid(x) #the first non-linear activation function, for others like ReLU use torch.nn.functional.relu_()
        x = x.view(-1,self.d2,self.d3) # b*d2*d3
   
        G_mat = torch.bmm(v,x) #b*1*d2 bmm b*d2*d3 = b*1*d3
        G_mat = torch.nn.functional.sigmoid(G_mat)# the second non-linear activation function
        G_mat = G_mat.view(-1,self.d3,1) #b*d3*1

        x = torch.bmm(w, G_mat) #b*1*d3 bmm b*d3*1=b*1*1
        pred = torch.squeeze(x)
        return pred