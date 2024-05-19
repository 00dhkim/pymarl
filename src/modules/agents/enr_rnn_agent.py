from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ENRRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(ENRRNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.mc_approx = args.mc_approx
        self.g = torch.Generator(device=args.device)
        self.n_random_layer = 0

        self.random_layers = [None] * self.mc_approx
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
    
    # initialized every peisode
    def init_random_layer(self, seeds: List[int]):
        # clean_flag일 떄에는 seed 하나만 주어짐
        self.n_random_layer = min(self.mc_approx, len(seeds))
        for i, seed in zip(range(self.mc_approx), seeds):
            self.g.manual_seed(seed)
            self.random_layers[i] = torch.diag(torch.FloatTensor(self.input_shape).uniform_(self.args.uniform_matrix_start, self.args.uniform_matrix_end, generator=self.g)).to(self.fc1.weight.device)

    def forward(self, clean_flag, inputs, hidden_state, idx_mc):
        
        clean_inputs = inputs.clone().detach()
        inputs = self._forward_random_layer(clean_flag, inputs, idx_mc)
        assert inputs.shape == clean_inputs.shape
        
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        
        return q, h

    def _forward_random_layer(self, clean_flag, inputs, idx_mc):
        if clean_flag:
            return inputs
        
        return torch.matmul(inputs, self.random_layers[idx_mc])