import torch
import torch.nn.functional as func
from torch_geometric.nn import LEConv, GATConv, TransformerConv, GCNConv, GraphConv
from torch_geometric.nn import global_mean_pool


class GNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim,layer_width):
        torch.manual_seed(123)
        super(GNN, self).__init__()
        self.input_layer = LEConv(input_dim, layer_width) #LEConv
        self.inner_layer = LEConv(layer_width,layer_width) #GraphConv
        self.output_layer = LEConv(layer_width, output_dim)
        # self.norm = torch.nn.LayerNorm(layer_width)
        # self.output_layer = LEConv(2 * layer_width, layer_width)
        # self.classifier = torch.nn.Linear(layer_width, output_dim)
        # self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, data,number_of_layers):
        x, edge_index,edge_weight = data.x.float(), data.edge_index, data.edge_attr.float()
        x = func.leaky_relu(self.input_layer(x, edge_index,edge_weight),negative_slope = 0.1)
        # x = self.dropout(x)
        # x = self.norm(x)
        for i in range(number_of_layers-2):
            x = func.leaky_relu(self.inner_layer(x,edge_index,edge_weight),negative_slope = 0.1)
            # x = self.dropout(x)
            # x = self.norm(x)  
        #Add global graph context here
        # global_feat = global_mean_pool(x, data.batch)        # [num_graphs, hidden_dim]
        # global_feat = global_feat[data.batch]                # [num_nodes, hidden_dim]
        # x = torch.cat([x, global_feat], dim=1)               # [num_nodes, 2*hidden_dim]
        x = func.leaky_relu(self.output_layer(x,edge_index,edge_weight),negative_slope = 0.1)
        
        #Classifier layer
        # x = self.classifier(x)                               # e.g., Linear(2*hidden_dim, 1)
        x = torch.sigmoid(x)
        return x  

class EarlyStopping: # https://www.geeksforgeeks.org/how-to-handle-overfitting-in-pytorch-models-using-early-stopping/
    def __init__(self, patience=50, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)