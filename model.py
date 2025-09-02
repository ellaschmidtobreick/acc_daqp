import torch
import torch.nn.functional as func
from torch_geometric.nn import LEConv,GCNConv

class GNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim,layer_width):
        torch.manual_seed(123)
        super(GNN, self).__init__()
        self.input_layer = LEConv(input_dim, layer_width)
        self.inner_layer = LEConv(layer_width,layer_width)
        self.output_layer = LEConv(layer_width, output_dim)

    def forward(self, data,number_of_layers):
        x, edge_index,edge_weight = data.x.float(), data.edge_index, data.edge_attr.float()
        x = func.leaky_relu(self.input_layer(x, edge_index,edge_weight),negative_slope = 0.1)
        for i in range(number_of_layers-2):
            x = func.leaky_relu(self.inner_layer(x,edge_index,edge_weight),negative_slope = 0.1)
        #x = func.leaky_relu(self.output_layer(x,edge_index,edge_weight),negative_slope = 0.1)
        x = self.output_layer(x,edge_index,edge_weight)
        # print("x", x)
        x = torch.sigmoid(x)
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