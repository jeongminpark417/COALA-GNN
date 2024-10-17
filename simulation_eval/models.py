import torch.nn as nn
import torch.nn.functional as F
from dgl import apply_each
from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv, HeteroGraphConv
import dgl.nn.pytorch as dglnn


class DistSAGE(nn.Module):
    """
    SAGE model for distributed train and evaluation.

    Parameters
    ----------
    in_feats : int
        Feature dimension.
    n_hidden : int
        Hidden layer dimension.
    n_classes : int
        Number of classes.
    n_layers : int
        Number of layers.
    activation : callable
        Activation function.
    dropout : float
        Dropout value.
    """

    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout=0.2
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        for _ in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        """
        Forward function.

        Parameters
        ----------
        blocks : List[DGLBlock]
            Sampled blocks.
        x : DistTensor
            Feature data.
        """
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

class SAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers=2, dropout=0.2):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, h_feats, aggregator_type='mean'))
        for _ in range(num_layers-2):
            self.layers.append(SAGEConv(h_feats, h_feats, aggregator_type='mean'))
        self.layers.append(SAGEConv(h_feats, num_classes, aggregator_type='mean'))
        self.dropout = nn.Dropout(dropout)
    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.num_dst_nodes()]
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.dropout(h)
                h = F.relu(h)


        return h

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers=2, dropout=0.2):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, h_feats))
        for _ in range(num_layers-2):
            self.layers.append(GraphConv(h_feats, h_feats))
        self.layers.append(GraphConv(h_feats, num_classes))
        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.num_dst_nodes()]
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.dropout(h)
                h = F.relu(h)
        return h

# class GAT(nn.Module):
#     def __init__(self, in_feats, h_feats, num_classes, num_heads, num_layers=2, dropout=0.2):
#         super(GAT, self).__init__()
#         self.layers = nn.ModuleList()
#         self.layers.append(GATConv(in_feats, h_feats, num_heads))
#         for _ in range(num_layers-2):
#             self.layers.append(GATConv(h_feats * num_heads, h_feats, num_heads))
#         self.layers.append(GATConv(h_feats * num_heads, num_classes, num_heads))
#         self.dropout = nn.Dropout(dropout)

#    def forward(self, blocks, x):
#        h = x
#        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
#            h_dst = h[:block.num_dst_nodes()]
#            if l < len(self.layers) - 1:
#                h = layer(block, (h, h_dst)).flatten(1)
#                h = F.relu(h)
#                h = self.dropout(h)
#            else:
#                h = layer(block, (h, h_dst)).mean(1)  
#        return h

class GAT(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, num_heads
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.GATConv(
                (in_feats, in_feats),
                n_hidden,
                num_heads=num_heads
            )
        )
        for i in range(1, n_layers - 1):
            self.layers.append(
                dglnn.GATConv(
                    (n_hidden * num_heads, n_hidden * num_heads),
                    n_hidden,
                    num_heads=num_heads
                )
            )
        self.layers.append(
            dglnn.GATConv(
                (n_hidden * num_heads, n_hidden * num_heads),
                n_classes,
                num_heads=num_heads,
                activation=None,
            )
        )

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[: block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            if l < self.n_layers - 1:
                h = layer(block, (h, h_dst)).flatten(1)
            else:
                h = layer(block, (h, h_dst))
        h = h.mean(1)
        return h.log_softmax(dim=-1)



class RGCN(nn.Module):
    def __init__(self, etypes, in_feats, h_feats, num_classes, num_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HeteroGraphConv({
            etype: GraphConv(in_feats, h_feats)
            for etype in etypes}, aggregate='mean'))
        for _ in range(num_layers-2):
            self.layers.append(HeteroGraphConv({
                etype: GraphConv(h_feats, h_feats)
                for etype in etypes}, aggregate='mean'))
        self.layers.append(HeteroGraphConv({
            etype: GraphConv(h_feats, h_feats)
            for etype in etypes}, aggregate='mean'))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_feats, num_classes)  

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1]))
            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)
        return self.linear(h['paper'])

class RSAGE(nn.Module):
    def __init__(self, etypes, in_feats, h_feats, num_classes, num_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HeteroGraphConv({
            etype: SAGEConv(in_feats, h_feats, 'gcn')
            for etype in etypes}))
        for _ in range(num_layers-2):
            self.layers.append(HeteroGraphConv({
                etype: SAGEConv(h_feats, h_feats, 'gcn')
                for etype in etypes}))
        self.layers.append(HeteroGraphConv({
            etype: SAGEConv(h_feats, h_feats, 'gcn')
            for etype in etypes}))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_feats, num_classes)  

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1]))
            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)
        return self.linear(h['paper'])


# class GAT(nn.Module):
#     def __init__(
#         self, in_feats, n_hidden, n_classes, n_layers, num_heads
#     ):
#         super().__init__()
#         self.n_layers = n_layers
#         self.n_hidden = n_hidden
#         self.n_classes = n_classes
#         self.layers = nn.ModuleList()
#         self.layers.append(
#             dglnn.GATConv(
#                 (in_feats, in_feats),
#                 n_hidden,
#                 num_heads=num_heads
#             )
#         )
class RGAT(nn.Module):
    def __init__(self, etypes, in_feats, h_feats, num_classes, num_layers=2, n_heads=4, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HeteroGraphConv({
            etype: GATConv(in_feats, h_feats // n_heads, n_heads)
            for etype in etypes}))
        for _ in range(num_layers-2):
            self.layers.append(HeteroGraphConv({
                etype: GATConv(h_feats, h_feats // n_heads, n_heads)
                for etype in etypes}))
        self.layers.append(HeteroGraphConv({
            etype: GATConv(h_feats, h_feats // n_heads, n_heads)
            for etype in etypes}))

     
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_feats, num_classes)


    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)
        return self.linear(h['paper'])   


      
