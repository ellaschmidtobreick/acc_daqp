import torch
import torch.nn.functional as func
from torch.nn import init 
from torch_geometric.nn import LEConv,GCNConv,GATConv

class GNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim,layer_width,conv_type="LEConv"):
        torch.manual_seed(123)
        super(GNN, self).__init__()
        
        if conv_type == "LEConv":
            self.input_layer = LEConv(input_dim, layer_width)
            self.inner_layer = LEConv(layer_width,layer_width)
            self.output_layer = LEConv(layer_width, output_dim)
            self.norm1 = torch.nn.LayerNorm(layer_width)
            self.norm2 = torch.nn.LayerNorm(layer_width)
        if conv_type == "GCN":
            self.input_layer = GCNConv(input_dim, layer_width)
            self.inner_layer = GCNConv(layer_width,layer_width)
            self.output_layer = GCNConv(layer_width, output_dim)
            self.norm1 = torch.nn.LayerNorm(layer_width)
            self.norm2 = torch.nn.LayerNorm(layer_width)
        if conv_type == "GAT":
            self.input_layer = GATConv(input_dim, layer_width, heads=4, concat=False)
            self.inner_layer = GATConv(layer_width, layer_width, heads=4, concat=False)
            self.output_layer = GATConv(layer_width, output_dim, heads=4, concat=False)
            self.norm1 = torch.nn.LayerNorm(layer_width)
            self.norm2 = torch.nn.LayerNorm(layer_width)

    def forward(self, data,number_of_layers,conv_type):
        x, edge_index,edge_weight = data.x.float(), data.edge_index, data.edge_attr.float()
        # print("x before standardized",x)
        # standardize input
        x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
        # print("x after standardization",x)
        # only positive edge weights
        if conv_type == "GCN" or conv_type == "GAT":
            edge_weight = edge_weight - edge_weight.min() + 1e-6  

        x = func.leaky_relu(self.norm1(self.input_layer(x, edge_index,edge_weight)),negative_slope = 0.1) #0001)
        # print("x after first layer",x)
        for i in range(number_of_layers-2):
            x = func.leaky_relu(self.norm2(self.inner_layer(x,edge_index,edge_weight)),negative_slope = 0.1) #0001)
        # print("x after inner layers",x)
        x = self.output_layer(x,edge_index,edge_weight)
        # print("x before sigmoid",x)
        # x = x / 2
        x = torch.sigmoid(x)
        # print("x after sigmoid",x)
        return x  
    
    def init_weights(self, p_pos=0.05): 
        gain = init.calculate_gain('leaky_relu', param=0.0001)

        for layer in [self.input_layer, self.inner_layer]:
            if hasattr(layer, 'lin'):
                init.xavier_uniform_(layer.lin.weight, gain=gain)
                if layer.lin.bias is not None:
                    init.zeros_(layer.lin.bias)

        # Output layer initialization
        if hasattr(self.output_layer, 'lin'):
            init.xavier_uniform_(self.output_layer.lin.weight, gain=1.0)

            # bias to reflect class imbalance
            bias_value = torch.log(torch.tensor(p_pos / (1 - p_pos)))
            self.output_layer.lin.bias.data.fill_(bias_value)



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
        self.norm1 = torch.nn.LayerNorm(layer_width)
        self.norm2 = torch.nn.LayerNorm(layer_width)

    def forward(self, x):
        x = func.leaky_relu(self.norm1(self.input_layer(x)),negative_slope = 0.0001) #0.1
        x = func.leaky_relu(self.norm2(self.hidden_layer(x)),negative_slope = 0.0001)
        return torch.sigmoid(self.output_layer(x)) 