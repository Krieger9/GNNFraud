import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import HeteroConv, GATConv, Linear, SAGEConv, to_hetero
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Dropout
from torch.nn.functional import relu, tanh, softmax
import torch_geometric.transforms as T

class EnhancedGATModel(torch.nn.Module):
    @property
    def edge_types(self):
        return [
            ('user', 'owns', 'card'),
            ('card','belongs_to','user'),
            ('user', 'has', 'user_history'),
            ('user_history', 'belongs_to', 'user'),
            ('user_history_transaction', 'part_of', 'user_history'),
            ('user_history_transaction', 'paid_with', 'card'),
            ('user_history_transaction', 'made_at', 'merchant'),
            ('card','paid_for','user_history_transaction'),
            ('merchant', 'made', 'user_history_transaction'),
            ('user_history', 'reflects_on', 'pending_transaction'),            
            ('merchant', 'made', 'user_history_transaction'),
            ('merchant', 'selling', 'pending_transaction'),
            ('user', 'purchasing', 'pending_transaction')
        ]

    def __init__(self, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: GATConv((-1, -1), hidden_channels, add_self_loops=False)
                for edge_type in self.edge_types
            }, aggr='mean')
            self.convs.append(conv)
            self.dropouts.append(Dropout(dropout_rate)) 
        
        self.lin1 = Linear(3*hidden_channels, hidden_channels * 6)
        self.lin2 = Linear(hidden_channels * 6, out_channels)
        #self.skip_lin = Linear(105, 3*hidden_channels)
        #self.bn_skip=nn.LayerNorm(105)
        self.dropout_1 = Dropout(dropout_rate)
        self.dropout_2 = Dropout(dropout_rate)

    def forward(self, x_dict, edge_index_dict):
        #x_pending_transaction = x_dict['pending_transaction']
        #x_user = x_dict['user']
        #x_transaction_history=x_dict['user_history']
        #x_skip = torch.cat([x_pending_transaction, x_user, x_transaction_history], dim=1)

        for conv, dropout in zip(self.convs, self.dropouts):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: tanh(dropout(x)) for key, x in x_dict.items()}
        
        x_pending_transaction = x_dict['pending_transaction']
        x_user = x_dict['user']
        x_transaction_history = x_dict['user_history']
        #print((x_pending_transaction.shape, x_user.shape, x_transaction_history.shape, x_skip.shape))
        
        combined = torch.cat([x_pending_transaction, x_user, x_transaction_history], dim=1)        
        combined = self.dropout_1(combined)
        combined = tanh(self.lin1(combined))        
        
        out = self.lin2(combined)
        return out