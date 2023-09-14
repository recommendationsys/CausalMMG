from torch import nn
from torch.nn import Parameter
import torch
import math
from torch.nn import functional as F



class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, config,in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.use_cuda = config['use_cuda']
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features)).to(self.device)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features)).to(self.device)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight).to(self.device)
        output = torch.spmm(adj.to(self.device), support).to(self.device)
        if self.bias is not None:
            return F.relu(output + self.bias)
        else:
            return F.relu(output)

    def norm(self, adj, symmetric=True):
        # A = A+I
        new_adj = adj + torch.eye(adj.size(0)).to(tt.arg.device)
        degree = new_adj.sum(1)
        if symmetric:
            # degree = degree^-1/2
            degree = torch.diag(torch.pow(degree, -0.5))
            return degree.mm(new_adj).mm(degree)
        else:
            # degree=degree^-1
            degree = torch.diag(torch.pow(degree, -1))
            return degree.mm(new_adj)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'