import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from torch.distributions import Normal, kl_divergence as kl

# zinb_loss
def zinb_log_positive(x, mu, theta, pi, eps=1e-8):
    x=x.float()

    if theta.ndimension()==1: # tensor's dimension
        theta=theta.view(1, theta.size(0))

    softplus_pi=F.softplus(-pi)
    log_theta_eps=torch.log(theta + eps)
    log_theta_mu_eps=torch.log(theta + mu + eps)
    pi_theta_log=-pi + theta * (log_theta_eps - log_theta_mu_eps)
    case_zero=F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero=torch.mul((x < eps).type(torch.float32), case_zero)
    case_non_zero=( -softplus_pi + pi_theta_log + x * ( torch.log( mu + eps )-log_theta_mu_eps) + torch.lgamma( x + theta ) - torch.lgamma(theta) - torch.lgamma( x + 1 ))
    mul_case_non_zero=torch.mul( (x>eps).type(torch.float32), case_non_zero)
    res = mul_case_zero+mul_case_non_zero

    return -torch.sum( res, dim = 1 )

# A_rec_loss=tf.reduce_mean(MSE(self.adj, A_out))
def mse_loss(y_true, y_pred):

    mask = torch.sign(y_true)

    y_pred = y_pred.float()
    y_true = y_true.float()

    ret = torch.pow( (y_pred - y_true) * mask , 2)

    return torch.sum( ret, dim = 1 )

def A_recloss(adj, A_out):
    A_rec_loss=torch.mean(mse_loss(adj, A_out))
    return A_rec_loss

def cluster_loss(q_out, p):
    loss = torch.mean(F.kl_div(q_out.log(), p))
    return loss


def distribution_loss(Q, P):
    """
    calculate the clustering guidance loss L_{KL}
    Args:
        Q: the soft assignment distribution
        P: the target distribution
    Returns: L_{KL}
    """
    loss = F.kl_div((Q[0].log() + Q[1].log() + Q[2].log()) / 3, P, reduction='batchmean')
    return loss
