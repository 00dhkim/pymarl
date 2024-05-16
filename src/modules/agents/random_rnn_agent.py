import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RandomRNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.mc_approx = args.mc_approx

        self.random_layers = [None] * self.mc_approx
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
    
    # initialized every peisode
    def init_random_layer(self):
        for i in range(self.mc_approx):
            self.random_layers[i] = torch.diag(torch.FloatTensor(self.input_shape).uniform_(self.args.uniform_matrix_start, self.args.uniform_matrix_end)).to(self.fc1.weight.device)

    def forward(self, clean_flag, inputs, hidden_state, idx_mc):
        
        if not clean_flag:
            clean_inputs = inputs
            inputs = torch.matmul(inputs, self.random_layers[idx_mc])
            assert inputs.shape == clean_inputs.shape
        
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        
        return q, h
