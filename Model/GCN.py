import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        
    def forward(self, x, adj):
        x = self.linear(torch.spmm(adj, x))  # graph convolution layer
        x = self.bn(x)
        return x

class GCN(nn.Module):
    def __init__(self, 
                 nfeat=1433,        # input feature dimension
                 layer_dims=[64, 64], 
                 nclass=7,          # output class dimension
                 dropout=0.5):
        super(GCN, self).__init__()
        layers = []
        in_dim = nfeat
        
        # hidden layers
        for dim in layer_dims[:-1]:
            layers.append(GCNLayer(in_dim, dim))
            in_dim = dim
            
        # output layer
        self.layers = nn.ModuleList(layers)
        self.out_layer = GCNLayer(in_dim, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return self.out_layer(x, adj)