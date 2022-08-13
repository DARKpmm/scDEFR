from re import A
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, Parameter
from sigmalayer import Encoder, decoder_ZINB
# readout function
class Readout(nn.Module):
    def __init__(self, K):
        super(Readout, self).__init__()
        self.K = K

    def forward(self, Z):
        # calculate cluster-level embedding
        Z_tilde = []

        # step1: split the nodes into K groups
        # step2: average the node embedding in each group
        n_node = Z.shape[0]
        step = n_node // self.K
        for i in range(0, n_node, step):
            if n_node - i < 2 * step:
                Z_tilde.append(torch.mean(Z[i:n_node], dim=0))
                break
            else:
                Z_tilde.append(torch.mean(Z[i:i + step], dim=0))

        # the cluster-level embedding
        Z_tilde = torch.cat(Z_tilde, dim=0)
        return Z_tilde.view(1, -1)

class DCRN(nn.Module):
  def __init__(self, args, layerd, hidden, zdims, dropout, G, n_node):
      super(DCRN, self).__init__()
      self.sigma = args.sigma
      self.Lencoder=Encoder(args.n1, args.n2, args.n3, args.n4, dropout, n_clusters=args.n_clusters)
      self.Gencoder=Encoder(args.n1, args.n2, args.n3, args.n4, dropout, g=G, n_clusters=args.n_clusters)

      self.Decoder=decoder_ZINB(layerd, hidden, args.n4, dropout)
      # fusion parameter
      self.a = Parameter(nn.init.constant_(torch.zeros(n_node, zdims), 0.5), requires_grad=True)
      self.b = Parameter(nn.init.constant_(torch.zeros(n_node, zdims), 0.5), requires_grad=True)
      self.alpha = Parameter(torch.zeros(1))

      self.cluster_centers = Parameter(torch.Tensor(args.n_clusters, zdims), requires_grad=True) 

      self.R = Readout(K=args.n_clusters)

  def q_distribution(self, z, z_l, z_g):
    q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_centers, 2), 2))
    q = (q.t() / torch.sum(q, 1)).t()

    q_l = 1.0 / (1.0 + torch.sum(torch.pow(z_l.unsqueeze(1) - self.cluster_centers, 2), 2))
    q_l = (q_l.t() / torch.sum(q_l, 1)).t()

    q_g = 1.0 / (1.0 + torch.sum(torch.pow(z_g.unsqueeze(1) - self.cluster_centers, 2), 2))
    q_igae = (q_g.t() / torch.sum(q_g, 1)).t()

    return q, q_l, q_igae
    

    


  def forward(self, X, Am):
    
    #(1 - sigma) * h + sigma * tra1
    relu=nn.ReLU()
    enc_h1_g = self.Gencoder.forward_g1(X)
    enc_h1_l = self.Lencoder.dropout(relu(self.Lencoder.BN1(self.Lencoder.enc1(X))))
    enc_h2_g = self.Gencoder.forward_g2((1-self.sigma)*enc_h1_g+self.sigma*enc_h1_l)
    enc_h2_l = self.Lencoder.dropout(relu(self.Lencoder.BN2(self.Lencoder.enc2((1-self.sigma)*enc_h1_g+self.sigma*enc_h1_l))))
    enc_h3_g = self.Gencoder.forward_g3((1-self.sigma)*enc_h2_g+self.sigma*enc_h2_l)
    enc_h3_l = self.Lencoder.dropout(relu(self.Lencoder.BN3(self.Lencoder.enc3((1-self.sigma)*enc_h2_g+self.sigma*enc_h2_l))))
    z_L= enc_h3_l
    z_G= enc_h3_g
    A_pred = self.Gencoder.dot_product_decoder(z_G)
    #pi_g, disp_g, mean_g = self.Decoder(z_g)
    #pi_l, disp_l, mean_l = self.Decoder(z_l)

    #node embedding fusion
    z_i = self.a*z_L +self.b*z_G
    #.double()
    z_l = torch.spmm(Am, z_i.double())
    s = torch.mm(z_l, z_l.t())
    s = F.softmax(s, dim=1)
    z_g = torch.mm(s, z_l)
    z = self.alpha*z_g+z_l
    #pred_l=self.Lencoder.enc4(z.float())

    pi, disp, mean = self.Decoder(z)
    self.mean=mean

    # soft assignment distribution q
    Q, QL, QG = self.q_distribution(z, z_L, z_G)

    return z, Q, QL, QG, A_pred, pi, disp, mean