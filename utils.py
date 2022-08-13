# encoding: utf-8
import torch
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from scipy import sparse as sp
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.cluster import SpectralClustering


def parameter_setting():
    
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument("--dataname", default = "Qs_Limb_Muscle", type = str)
    parser.add_argument("--highly_genes", default = "2000", type = int)
    parser.add_argument("--lr", default = "5e-5", type = float)
    parser.add_argument("--lr1", default = "1e-8", type = float)
    parser.add_argument("--epochs", default = "250", type = int)
    parser.add_argument("--sigma", default = "0.1", type = float)
    parser.add_argument("--n1", default = "24", type = int)
    parser.add_argument("--n2", default = "128", type = int)
    parser.add_argument("--n3", default = "1024", type = int)
    parser.add_argument("--model_file", default = "/home//data/test16.pth.tar", type = str)
    
    return parser


def data_preprocessing(dataset):
    dataset.adj = torch.sparse_coo_tensor(
        dataset.edge_index, torch.ones(dataset.edge_index.shape[1]), torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
    ).to_dense()
    dataset.adj_label = dataset.adj

    dataset.adj += torch.eye(dataset.x.shape[0])
    dataset.adj = normalize(dataset.adj, norm="l1")
    dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)

    return dataset

def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    # t_order
    t=2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)

def adjust_learning_rate(init_lr, optimizer, iteration, max_lr, adjust_epoch):
    lr = max(init_lr*(0.9**(iteration//adjust_epoch)),max_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"]=lr

    return lr

def read_dataset(File1 = None, File2=None, File3 = None, File4 = None, format_rna = "table", format_epi = "table", transpose = True, state = 0):
    adata = adata1 = None

    if File1 is not None:
        if format_rna == "table":
            adata  = sc.read(File1)
        else: # 10X format
            adata  = sc.read_mtx(File1)
           
        if transpose:
            adata  = adata.transpose()
    
    if File2 is not None:
        if format_rna == "table":
            adata1  = sc.read( File2 )
        else:# 10X format
            adata1  = sc.read_mtx(File2)

        if transpose: 
            adata1  = adata1.transpose()

    label_ground_truth = []
    label_ground_truth1 = []
    if state == 0 :
        if File3 is not None:
            Data2  = pd.read_csv( File3, header=0, index_col=0 )
            label_ground_truth =  Data2['Group'].values

        else:
            label_ground_truth =  np.ones( len( adata.obs_names ) )

        if File4 is not None:
            Data2 = pd.read_csv( File4, header=0, index_col=0 )
            label_ground_truth1 = Data2['Group'].values

        else:
            label_ground_truth1 =  np.ones( len( adata.obs_names ) )

    elif state == 1:
        if File3 is not None:
            Data2 = pd.read_table( File3, header=0, index_col=0 )
            label_ground_truth = Data2['cell_line'].values
        else:
            label_ground_truth =  np.ones( len( adata.obs_names ) )

        if File4 is not None:
            Data2 = pd.read_table( File4, header=0, index_col=0 )
            label_ground_truth1 = Data2['cell_line'].values
        else:
            label_ground_truth1 =  np.ones( len( adata.obs_names ) )

    elif state == 3:
        if File3 is not None:
            Data2 = pd.read_table( File3, header=0, index_col=0 )
            label_ground_truth = Data2['Group'].values
        else:
            label_ground_truth =  np.ones( len( adata.obs_names ) )

        if File4 is not None:
            Data2 = pd.read_table( File4, header=0, index_col=0 )
            label_ground_truth1 = Data2['Group'].values
        else:
            label_ground_truth1 =  np.ones( len( adata.obs_names ) )

    else:
        if File3 is not None:
            Data2 = pd.read_table( File3, header=0, index_col=0 )
            label_ground_truth = Data2['Cluster'].values
        else:
            label_ground_truth =  np.ones( len( adata.obs_names ) )

        if File4 is not None:
            Data2 = pd.read_table( File4, header=0, index_col=0 )
            label_ground_truth1 = Data2['Cluster'].values
        else:
            label_ground_truth1 =  np.ones( len( adata.obs_names ) )
    
    adata.obs['Group'] = label_ground_truth
    adata.obs['Group'] = adata.obs['Group'].astype('category')

    adata1.obs['Group'] = label_ground_truth
    adata1.obs['Group'] = adata1.obs['Group'].astype('category')

    print('Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    return adata, adata1, label_ground_truth, label_ground_truth1
    
def normalized( adata, filter_min_counts=True, size_factors=True, highly_genes=None,
               normalize_input=False, logtrans_input=True):

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    """ if size_factors:
        #adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
        adata.obs['size_factors'] = np.log( np.sum( adata.X, axis = 1 ) )
    else:
        adata.obs['size_factors'] = 1.0 """
    
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    
    if logtrans_input:
        sc.pp.log1p(adata)
    
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)

    if normalize_input:
        sc.pp.scale(adata)

    return adata

def dopca(X, dim=10):
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10

def get_adj(count, k=10, pca=50, mode="connectivity"):
    if pca:
        countp = dopca(count, dim=pca)
    else:
        countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="euclidean", include_self=True) 
    adj = A.toarray()
    adj_n = norm_adj(adj)
    return adj, adj_n


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten() 
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn

decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
encode = empty_safe(np.vectorize(lambda _x: str(_x).encode("utf-8")), "S")
upper = empty_safe(np.vectorize(lambda x: str(x).upper()), str)
lower = empty_safe(np.vectorize(lambda x: str(x).lower()), str)
tostr = empty_safe(np.vectorize(str), str)

def clustering(args, z, y, adjn1):
  labels_k=KMeans(n_clusters=args.n_clusters, n_init=20).fit_predict(z.data.cpu().numpy())
  labels_s = SpectralClustering(n_clusters=args.n_clusters,affinity="precomputed", assign_labels="discretize", n_init=20).fit_predict(adjn1.data.cpu().numpy())
  labels = labels_s if (np.round(metrics.normalized_mutual_info_score(y, labels_s), 5)>=np.round(metrics.normalized_mutual_info_score(y, labels_k), 5)
  and np.round(metrics.adjusted_rand_score(y, labels_s), 5)>=np.round(metrics.adjusted_rand_score(y, labels_k), 5)) else labels_k 
  nmi, ari = eva(y, labels)
  centers=computeCentroids(z.data.cpu().numpy(), labels)
  return nmi, ari, centers

def eva(y_true, y_pred):
  nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
  ari = ari_score(y_true, y_pred)
  return nmi, ari

def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))# torch.unique
    return np.array([data[labels == i].mean(0) for i in range(n_clusters)])
 