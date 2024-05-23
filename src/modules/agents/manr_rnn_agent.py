import torch
import torch.nn as nn
import torch.nn.functional as F


class ManrRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(ManrRNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.mc_approx = args.mc_approx
        self.n_agents = args.n_agents

        # 에이전트와 Monte Carlo 샘플을 위한 random layers 초기화
        self.random_layers = nn.ParameterList([
            nn.Parameter(
                torch.diag(
                    torch.FloatTensor(input_shape).uniform_(
                        self.args.uniform_matrix_start, self.args.uniform_matrix_end)
                ), requires_grad=False
            ) for _ in range(self.n_agents * self.mc_approx)
        ])

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self, batch_size):
        # return self.fc1.weight.new(batch_size, self.n_agents, self.args.rnn_hidden_dim).zero_()
        return self.fc1.weight.new(batch_size, self.args.rnn_hidden_dim).zero_()

    def forward(self, clean_flag, inputs, hidden_state, idx_mc):
        # inputs: (batch_size, n_agents, input_shape)
        
        if not clean_flag:
            clean_inputs = inputs.clone().detach()
            for i in range(self.n_agents):
                inputs[:, i, :] = torch.matmul(inputs[:, i, :], self.random_layers[i * self.mc_approx + idx_mc])
            assert inputs.shape == clean_inputs.shape

        x = F.relu(self.fc1(inputs.view(-1, self.input_shape))).view(inputs.shape[0], self.n_agents, -1)
        h_in = hidden_state.view(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x.view(-1, self.args.rnn_hidden_dim), h_in)
        h = h.view(inputs.shape[0], self.n_agents, -1)
        q = self.fc2(h.view(-1, self.args.rnn_hidden_dim)).view(inputs.shape[0], self.n_agents, -1)
        
        return q, h
