import torch

class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        num_nodes = 256
        self.fc1 = torch.nn.Linear(input_dim, num_nodes)
        self.fc2 = torch.nn.Linear(num_nodes, num_nodes)
        self.fc3 = torch.nn.Linear(num_nodes, num_nodes)
        self.output = torch.nn.Linear(num_nodes, output_dim)

    def forward(self, input):
        act1 = torch.nn.functional.mish(self.fc1(input))
        act2 = torch.nn.functional.mish(self.fc2(act1))
        act3 = torch.nn.functional.mish(self.fc3(act2))
        output = self.output(act3)
        return output

class MLP_Qnetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_Qnetwork, self).__init__()
        num_nodes = 256
        self.fc1 = torch.nn.Linear(input_dim, num_nodes)
        self.fc2 = torch.nn.Linear(num_nodes, num_nodes)
        # self.fc3 = torch.nn.Linear(num_nodes, num_nodes)
        self.output = torch.nn.Linear(num_nodes, output_dim)

    def forward(self, input):
        act1 = torch.nn.functional.relu(self.fc1(input))
        act2 = torch.nn.functional.relu(self.fc2(act1))
        # act3 = torch.nn.functional.relu(self.fc3(act2))
        output = self.output(act2)
        return output

class CNN_1D(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN_1D, self).__init__()
        num_nodes = 256
        kernel_size = 3
        self.conv1 = torch.nn.Conv1d(input_dim, num_nodes, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = torch.nn.Conv1d(num_nodes, num_nodes, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv3 = torch.nn.Conv1d(num_nodes, num_nodes, kernel_size=kernel_size, padding=kernel_size//2)
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.output = torch.nn.Linear(num_nodes, output_dim)

    def forward(self, input):
        act1 = torch.nn.functional.mish(self.conv1(input))
        act2 = torch.nn.functional.mish(self.conv2(act1))
        act3 = torch.nn.functional.mish(self.conv3(act2))
        pool = self.global_pool(act3).squeeze(-1)
        output = self.output(pool)
        return output

class LNResNet(torch.nn.Module):
    def __init__(self, input_dim:int, output_dim:int, 
                 dropout_rate:float=0.00, layer_norm_use:bool=False, 
                 num_nodes:int=256, num_blocks:int=3):
        super(LNResNet, self).__init__()
        self.num_nodes = num_nodes
        self.dropout_rate = dropout_rate
        self.layer_norm_use = layer_norm_use
        self.num_blocks = num_blocks

        # Components
        self.fc1 = torch.nn.Linear(input_dim, num_nodes)
        # MLPResNet Components
        self.fc2 = torch.nn.Linear(num_nodes, num_nodes*4)
        self.fc3 = torch.nn.Linear(num_nodes*4, num_nodes)
        self.fc4 = torch.nn.Linear(num_nodes, num_nodes*4)
        self.fc5 = torch.nn.Linear(num_nodes*4, num_nodes)
        self.fc6 = torch.nn.Linear(num_nodes, num_nodes*4)
        self.fc7 = torch.nn.Linear(num_nodes*4, num_nodes)
        # Output layer
        self.output = torch.nn.Linear(num_nodes, output_dim)
    
    def MLPResNet(self, inputs):
        _data = inputs
        # 1st block
        if self.dropout_rate > 0:
            _data = torch.nn.Dropout(p=self.dropout_rate)(_data)
        if self.layer_norm_use:
            _data = torch.nn.LayerNorm()(_data)
        _data = self.fc2(_data)
        _data = torch.nn.functional.relu(_data)
        _data = self.fc3(_data)
        _data += inputs
        _1stblock_output = _data
        # 2nd block
        if self.dropout_rate > 0:
            _data = torch.nn.Dropout(p=self.dropout_rate)(_data)
        if self.layer_norm_use:
            _data = torch.nn.LayerNorm()(_data)
        _data = self.fc4(_data)
        _data = torch.nn.functional.relu(_data)
        _data = self.fc5(_data)
        _data += _1stblock_output
        _2ndblock_output = _data
        # 3rd block
        if self.dropout_rate > 0:
            _data = torch.nn.Dropout(p=self.dropout_rate)(_data)
        if self.layer_norm_use:
            _data = torch.nn.LayerNorm()(_data)
        _data = self.fc6(_data)
        _data = torch.nn.functional.relu(_data)
        _data = self.fc7(_data)
        _data += _2ndblock_output

        return _data

    def forward(self, inputs):
        _data = inputs
        _data = self.fc1(_data)
        _data = self.MLPResNet(_data)
        _data = torch.nn.functional.relu(_data)
        _data = self.output(_data)
        return _data