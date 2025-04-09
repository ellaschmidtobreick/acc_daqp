import torch
import torch.nn.functional as func
from torch_geometric.nn import LEConv

class GNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim,layer_width):
        torch.manual_seed(123)
        super(GNN, self).__init__()
        self.input = LEConv(input_dim, layer_width)
        #self.conv2 = LEConv(layer_width,layer_width)
        #self.conv3 = LEConv(layer_width,layer_width)
        self.inner = LEConv(layer_width,layer_width)
        self.output = LEConv(layer_width, output_dim)

    def forward(self, data,number_of_layers):
        x, edge_index,edge_weight = data.x.float(), data.edge_index, data.edge_attr.float()
        x = func.leaky_relu(self.input(x, edge_index,edge_weight),negative_slope = 0.1)
        for i in range(number_of_layers-2):
            x = func.leaky_relu(self.inner(x,edge_index,edge_weight),negative_slope = 0.1)
        #x = func.leaky_relu(self.conv2(x,edge_index,edge_weight),negative_slope = 0.1)
        #x = func.leaky_relu(self.conv3(x,edge_index,edge_weight),negative_slope = 0.1)
        #x = func.leaky_relu(self.conv4(x,edge_index,edge_weight),negative_slope = 0.1)
        x = func.leaky_relu(self.output(x,edge_index,edge_weight),negative_slope = 0.1)
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