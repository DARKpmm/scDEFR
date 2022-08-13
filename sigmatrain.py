from utils import normalized, get_adj, parameter_setting, clustering
from loss import zinb_log_positive, A_recloss, distribution_loss, cluster_loss
from sigmalayer import ZINB
from preprocess import prepro, normalize
from sigmamodel import DCRN

import os
import torch
import numpy as np
import scanpy as sc
import pandas as pd
from torch import optim
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import torch.nn.functional as F

def target_distribution(Q):
    """
    calculate the target distribution (student-t distribution)
    Args:
        Q: the soft assignment distribution
    Returns: target distribution P
    """
    weight = Q ** 2 / Q.sum(0)
    P = (weight.t() / weight.sum(1)).t()
    return P

def load_pretrain_parameter(model):
  pretrained_dict=torch.load(args.model_file)
  model_dict = model.state_dict()#
  pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
  model_dict.update(pretrained_dict)
  model.load_state_dict(model_dict)
  return model

def model_init(model, X, y, A_norm):
  model=load_pretrain_parameter(model)
  with torch.no_grad():
    z, _ , _, _, _, _, _, _=model(X, A_norm)
  _, _, centers=clustering(args, z, y, A_norm)
  return centers

def pre_train(model, args, X, y, adj):
  optimizer = optim.Adam( model.parameters(), lr = args.lr )
  model.train()
  for epoch in range(1, 1+args.epochs):
    
    z, Q, QL, QG, A_pred, pi, disp, mean=model(X, adj)
    
    rec_A_loss = A_recloss(adj, A_pred)

    zinb = ZINB(pi, theta=disp, ridge_lambda=0)
    loss_zinb = zinb.loss(X, mean, mean=True)
    loss = torch.mean(rec_A_loss+0.5*loss_zinb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch+=1
    if epoch%10==0:
      print(str(epoch)+ "   " + str(loss.item()) +"   "  + str(torch.mean(loss_zinb).item()) +"  "+
								   str(torch.mean(rec_A_loss).item()))
  model.eval()
  nmi, ari, _ = clustering(args, z, y, adj)
  torch.save(model.state_dict(), args.model_file)
  print("NMI: {:.4f},".format(nmi), "ARI: {:.4f},".format(ari))    
  print("pre_train over")

def alt_train(model, args, X, y, A_norm):
  centers=model_init(model, X, y, A_norm)
  model.cluster_centers.data=torch.tensor(centers).to("cuda:0")
  optimizer = optim.Adam( model.parameters(), lr = args.lr1 )
  for epoch in range(1, 1+args.epochs):
    
    z, Q, QL, QG, A_pred, pi, disp, mean = model(X, A_norm)
    
    rec_A_loss = A_recloss(A_norm, A_pred)
    
    zinb = ZINB(pi, theta=disp, ridge_lambda=0)
    loss_zinb = zinb.loss(X, mean, mean=True)
    
    loss_zinb=loss_zinb.double()

    L_clu=F.kl_div(Q.log(), target_distribution(Q), reduction='batchmean')
    L_gnn=F.binary_cross_entropy(QG, target_distribution(QG))
    L_dnn=F.binary_cross_entropy(QL, target_distribution(QL))

    loss = torch.mean(0.3*rec_A_loss+0.1*loss_zinb+0.5*L_clu+0.1*L_gnn+0.1*L_dnn)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch+=1

    if epoch%10==0:
      print(str(epoch)+ "   " + str(loss.item()) +"   "  + str(torch.mean(loss_zinb).item()) +"  "+
								   str(torch.mean(rec_A_loss).item()) +"  "+str( torch.mean(L_clu).item() ) +"  "+str( torch.mean(L_gnn).item() )
                   +" "+str(torch.mean(L_dnn).item()))
  nmi, ari, _ = clustering(args, z, y, A_norm)
  print(2)
  print("NMI: {:.4f},".format(nmi), "ARI: {:.4f},".format(ari))    
  print("alt_train over")
  return nmi, ari, z


def train_with_args(args):
  print(torch.__version__)
  args.highly_genes = 2000 
  args.lr = 5e-5
  args.lr1 = 1e-8
  args.epochs = 250
  args.sigma = 0.1 #0.7
  args.n1 = 24 #32
  args.n2 = 128
  args.n3 = 1024
  args.workdir    =  './Example_test/'
  args.path     =  './data/'
  args.model_file      = '/home//data/test16.pth.tar'
  
  x, y = prepro('/home//data/Qx_Limb_Muscle/data.h5')
  #x = np.array(pd.read_csv('/home//data/camp1/camp1.csv', header=None))
  #y = np.array(pd.read_csv('/home//data/camp1/camp1label.csv', header=None))
  x = np.ceil(x).astype(np.int)
  cluster_number = int(max(y) - min(y) + 1)   
  print(cluster_number) 
  args.n_clusters = cluster_number
  adata = sc.AnnData(x)
  adata.obs['Group'] = y
  adata  = normalize( adata, filter_min_counts=True, highly_genes=args.highly_genes,
                        size_factors=True, normalize_input=False, 
                        logtrans_input=True ) 
  print(adata)
  Nsample1, Nfeature   =  np.shape( adata.X )
  args.n4 = Nfeature
  adj1, adjn1 = get_adj(adata.X)
  A1 = coo_matrix(adj1)
  edge_index1 = torch.tensor([A1.row, A1.col], dtype=torch.long).to("cuda:0")
  X=torch.from_numpy(adata.X).to("cuda:0")
  adjn1=torch.from_numpy(adjn1).to("cuda:0")
  model=DCRN(args, layerd=[24, 128, 1024, Nfeature], hidden=args.n1, zdims=args.n1, dropout=0.01, G=edge_index1, n_node=X.shape[0]).to("cuda:0")
  #print(model)
  y=list(map(int, y))
  
  pre_train(model, args, X, y, adjn1)
  
  nmi, ari, z = alt_train(model, args, X, y, adjn1)
    
  print("NMI: {:.4f},".format(nmi), "ARI: {:.4f},".format(ari)) 

  

if __name__=="__main__":
  parser=parameter_setting()
  args = parser.parse_args()

  train_with_args(args)





  












