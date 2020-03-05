from layerGCN import GraphConvolution
import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy.special import digamma
import numpy as np
def torch_dirichlet_expectation(x):
    if len(x.size())==1:
        return  x.digamma()-x.sum().digamma()
    return x.digamma()-x.sum(dim=1, keepdim = True).digamma()
def dirichlet_expectation(alpha):
    if len(alpha.shape) == 1:
        return (digamma(alpha)-digamma(sum(alpha)))
    return (digamma(alpha)-digamma(np.sum(alpha, axis=1))[:, np.newaxis])
class GCTM(nn.Module):
    def __init__(self, V, num_topics, alpha, sigma, batchsize, iterate, hidden, nfeat, dropout):
        super(GCTM, self).__init__()
        self.V  = V
        self.K = num_topics
        self.alpha = alpha
        self.batchsize = batchsize
        self.hidden = hidden
        self.sigma = sigma
        self.gc1 = GraphConvolution(nfeat, self.hidden, bias=False)
        self.weightgc1=self.gc1.weight
        #self.weightgc1.register_hook(print)
        self.gc2 = GraphConvolution(self.hidden, self.K, bias=False)
        self.weightgc2 = self.gc2.weight
        #self.weightgc2.register_hook(print)
        self.dropout = dropout
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.sig = nn.Sigmoid()
        self.betat = nn.Parameter(torch.from_numpy(np.random.rand(self.K, self.V)).float())
        self.kappa = nn.Parameter(torch.from_numpy(np.ones((self.K,1))*0).float())
        self.iterate = iterate
    def computebeta(self, x, adj):
        z = F.relu(self.gc1(x, adj))
        z = F.dropout(z,self.dropout, training=self.training)
        z = self.gc2(z, adj)
        beta = torch.t(z)
        beta = F.sigmoid(self.kappa)*self.betat + (1-F.sigmoid(self.kappa))*beta
        return self.logsoftmax(beta)

    def updatebeta(self, x, adj):
        z = F.relu(self.gc1(x, adj))
        z = self.gc2(z, adj)
        beta = torch.t(z)
        beta = F.sigmoid(self.kappa)*self.betat + (1-F.sigmoid(self.kappa))*beta
        return self.logsoftmax(beta)

    def forward(self, inputs, x, adj, weightgc1, weightgc2, betat):
        logbeta = self.computebeta(x,adj)
        (gamma, total_phi) = self.inference(logbeta, inputs)
        return (logbeta, total_phi)

    def inference(self, logbeta, cnts):
        batchsize = cnts.size()[0]
        gamma = 1*np.random.gamma(100., 1./100., (batchsize,self.K ))
        ExpElogtheta = np.exp(dirichlet_expectation(gamma))
        betat = torch.t(torch.exp(logbeta.clone().detach())).numpy()
        total_phi = np.zeros(logbeta.size())
        for d in range(batchsize):
            cnt = cnts.narrow_copy(dim=0,start=d, length=1).to_dense()
            ids = torch.nonzero(cnt,as_tuple=True)[1].numpy()
            count = cnt.numpy()[0][ids]
            gammad = np.ones(self.K)*self.alpha+float(sum(count))/self.K
            ExpElogthetad = np.exp(dirichlet_expectation(gammad))
            betatd = betat[ids,:]
            for i in range(self.iterate):
                phi = ExpElogthetad*betatd +1e-10
                phi /=np.sum(phi,axis=1)[:,np.newaxis]
                gammad = self.alpha + np.dot(count,phi)
                ExpElogthetad = np.exp(dirichlet_expectation(gammad))
            gamma[d] = gammad
            ExpElogthetad = np.exp(dirichlet_expectation(gammad))
            phi = ExpElogthetad * betatd+1e-10
            phi /= np.sum(phi, axis=1)[:, np.newaxis]
            total_phi[:,ids] +=count*phi.transpose()
        return (torch.stack([torch.tensor(p, dtype=torch.float32) for p in gamma]), torch.stack([torch.tensor(p, dtype=torch.float32) for p in total_phi]))
    def mELBO(self,  total_phi, logbeta, weightgc1, weightgc2,betat):
        return -(total_phi*logbeta).sum() + (1/(2*self.sigma))*(self.gc1.weight - weightgc1).pow(2).sum() + (1/(2*self.sigma))*(self.gc2.weight - weightgc2).pow(2).sum() + (1/(2*self.sigma))*(self.betat - betat).pow(2).sum()