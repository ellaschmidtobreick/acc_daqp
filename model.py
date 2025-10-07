import torch
import torch.nn.functional as func
from torch_geometric.nn import LEConv,GCNConv,GATConv

class GNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim,layer_width,conv_type="LEConv"):
        torch.manual_seed(123)
        super(GNN, self).__init__()
        if conv_type == "LEConv":
            self.input_layer = LEConv(input_dim, layer_width)
            self.inner_layer = LEConv(layer_width,layer_width)
            self.output_layer = LEConv(layer_width, output_dim)
        if conv_type == "GCN":
            self.input_layer = GCNConv(input_dim, layer_width)
            self.inner_layer = GCNConv(layer_width,layer_width)
            self.output_layer = GCNConv(layer_width, output_dim)
        if conv_type == "GAT":
            self.input_layer = GATConv(input_dim, layer_width, heads=4, concat=False)
            self.inner_layer = GATConv(layer_width, layer_width, heads=4, concat=False)
            self.output_layer = GATConv(layer_width, output_dim, heads=4, concat=False)

    def forward(self, data,number_of_layers,conv_type):
        #print(data.x.min(), data.x.max())
        #data.x = (data.x - data.x.mean(dim=0)) / (data.x.std(dim=0) + 1e-6)
        x, edge_index,edge_weight = data.x.float(), data.edge_index, data.edge_attr.float()
        if conv_type == "GCN" or conv_type == "GAT":
            edge_weight = edge_weight - edge_weight.min() + 1e-6  # make all weights positive
        #print(x.min(), x.max())
            x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
        #print(x.min(), x.max())
        #print(torch.isnan(edge_weight).any(), edge_weight.min(), edge_weight.max())
        #print("x before conv:", torch.isnan(x).any(), x.min(), x.max())
        x = func.leaky_relu(self.input_layer(x, edge_index,edge_weight),negative_slope = 0.1)
        #print("x after conv:", torch.isnan(x).any(), x.min(), x.max())
        for i in range(number_of_layers-2):
            x = func.leaky_relu(self.inner_layer(x,edge_index,edge_weight),negative_slope = 0.1)
            #print(torch.isnan(x).any())
        #x = func.leaky_relu(self.output_layer(x,edge_index,edge_weight),negative_slope = 0.1)
        x = self.output_layer(x,edge_index,edge_weight)
        # print("x", x)
        x = torch.sigmoid(x)
        #print(torch.isnan(x).any())
        # print("x after sigmoid", x)
        return x  

# Define a simple GNN
# class GNN(torch.nn.Module):
#     def __init__(self,input_dim, output_dim,layer_width):
#         super().__init__()
#         self.input_layer = GCNConv(input_dim, layer_width)
#         self.inner_layer = GCNConv(layer_width, layer_width)
#         self.output_layer = GCNConv(layer_width, output_dim)
#         self.fc = torch.nn.Linear(1, 1)

#     def forward(self, data, number_of_layers):
#         x, edge_index, edge_weight= data.x, data.edge_index, data.edge_attr.float()
#         edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min() + 1e-6)
#         x = func.leaky_relu(self.input_layer(x, edge_index,edge_weight),negative_slope = 0.1)
#         for i in range(number_of_layers-2):
#             x = func.leaky_relu(self.inner_layer(x,edge_index,edge_weight),negative_slope = 0.1)
#         x = func.leaky_relu(self.output_layer(x,edge_index,edge_weight),negative_slope = 0.1)
#         return torch.sigmoid(x)  # Match y's shape
    
class EarlyStopping: # cite: https://www.geeksforgeeks.org/how-to-handle-overfitting-in-pytorch-models-using-early-stopping/
    def __init__(self, patience=50, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.epoch = epoch
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
        return self.epoch
    
class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim,layer_width=128):
        super().__init__()
        torch.manual_seed(123)
        self.input_layer = torch.nn.Linear(input_dim, layer_width)
        self.hidden_layer = torch.nn.Linear(layer_width, layer_width)
        self.output_layer = torch.nn.Linear(layer_width, output_dim)

    def forward(self, x):
        x = func.leaky_relu(self.input_layer(x),negative_slope = 0.1)
        x = func.leaky_relu(self.hidden_layer(x),negative_slope = 0.1)
        return torch.sigmoid(self.output_layer(x)) 