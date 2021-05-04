import torch
import torch.nn as nn
from learner.cx_graph_filter import CGraphFilter
import numpy as np


class HalfCxActor(nn.Module):

    def __init__(self, n_s, n_a, hidden_layers, k, ind_agg, device):
        """
        The policy network is allowed to have only one aggregation operation due to communication latency, but we can
        have any number of hidden layers to be executed by each agent individually.
        @param n_s: number of MDP states per agent
        @param n_a: number of MDP actions per agent
        @param hidden_layers: list of ints that will determine the width of each hidden layer
        @param k: aggregation filter length
        @param ind_agg: before which MLP layer index to aggregate
        @param device: CUDA device for torch
        """
        super(HalfCxActor, self).__init__()
        self.k = k
        self.n_s = n_s
        self.n_a = n_a
        self.hidden_layers = hidden_layers
        self.layers = [n_s] + hidden_layers + [n_a]
        self.n_layers = len(self.layers) - 1
        self.device = device
        self.ind_agg = ind_agg  # before which conv layer the aggregation happens

        self.cnn_layers = []
        self.conv_layers = []
        
        for i in range(0, self.n_layers):

            if i == self.ind_agg:  # after the GSO, reduce dimensions
                step = k
            else:
                step = 1
            
            # m = nn.Conv2d(in_channels=self.layers[i], out_channels=self.layers[i + 1], kernel_size=(step, 1),
            #               stride=(step, 1))

            # self.cnn_layers.append(m)
            
            # bn = nn.BatchNorm2d(self.layers[i + 1])
            # self.batch_norm.append(bn)
            cnn = nn.Conv2d(in_channels=self.layers[i], out_channels=self.layers[i + 1], kernel_size=(step, 1),
                          stride=(step, 1))
            if i < self.n_layers - 1:
                m = CGraphFilter(in_channels=self.layers[i], out_channels=self.layers[i + 1], stride=step, device=device)
            else:
                m = nn.Conv2d(in_channels=self.layers[i], out_channels=self.layers[i + 1], kernel_size=(step, 1),
                          stride=(step, 1))
                    
            self.conv_layers.append(m)
            self.cnn_layers.append(cnn)

        self.cnn_layers = torch.nn.ModuleList(self.cnn_layers)
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)


    def forward(self, delay_state, delay_gso, is_test):
        """
        The policy relies on delayed information from neighbors. During training, the full history for k time steps is
        necessary.
        @param delay_state: History of states: x_t, x_t-1,...x_t-k+1 of shape (B,K,F,N)
        @param delay_gso: Delayed GSO: [I, A_t, A_t A_t-1, A_t ... A_t-k+1] of shape (B,K,N,N)
        :return:
        """
        batch_size = delay_state.shape[0]
        n_agents = delay_state.shape[3]
        assert delay_gso.shape[0] == batch_size
        assert delay_gso.shape[2] == n_agents
        assert delay_gso.shape[3] == n_agents

        assert delay_state.shape[1] == self.k
        assert delay_state.shape[2] == self.n_s
        assert delay_gso.shape[1] == self.k

        x = delay_state
        x = x.permute(0, 2, 1, 3).to(self.device)  # now (B,F,K,N)
            
        for i in range(self.n_layers):
            if i == self.ind_agg:  # aggregation only happens once - otherwise each agent operates independently
                x = x.permute(0, 2, 1, 3)  # now (B,K,F,N)
                x = torch.matmul(x, delay_gso)
                x = x.permute(0, 2, 1, 3)  # now (B,F,K,N)

            if i < self.n_layers - 1:  # last layer - no activation
                x = self.conv_layers[i](x, is_test)  # now (B,G,N)
                # x = self.batch_norm[i](x)
                # if is_test:
                x = torch.sin(x)
                #x = F.relu(x)  # torch.tanh(x) #F.relu(x)
            else:
                x = self.conv_layers[i](x)
                # x = self.batch_norm[i](x)
            #     x = 10 * torch.tanh(x)
            
        # print("X " + str(x.shape))
        x = x.view((batch_size, 1, self.n_a, n_agents))  # now size (B, 1, nA, N)
        # x_norm = x - torch.mean(x, dim=0)
        # x = x.clamp(-10, 10)  # TODO these limits depend on the MDP
        
        x_gnn = delay_state
        x_gnn = x_gnn.permute(0, 2, 1, 3)  # now (B,F,K,N)

        for i in range(self.n_layers):

            if i == self.ind_agg:  # aggregation only happens once - otherwise each agent operates independently
                x_gnn = x_gnn.permute(0, 2, 1, 3)  # now (B,K,F,N)
                x_gnn = torch.matmul(x_gnn, delay_gso)
                x_gnn = x_gnn.permute(0, 2, 1, 3)  # now (B,F,K,N)

            x_gnn = self.cnn_layers[i](x_gnn)  # now (B,G,1,N)

            if i < self.n_layers - 1:  # last layer - no relu
                # x = self.layer_norms[i](x)
                x_gnn = torch.tanh(x_gnn)
                #x = F.relu(x)  # torch.tanh(x) #F.relu(x)
            # else:
            #     x = 10 * torch.tanh(x)

        x_gnn = x_gnn.view((batch_size, 1, self.n_a, n_agents))  # now size (B, 1, nA, N)

        return (x_gnn + 0.0001 * x)
            
    def project_filters(self):
        for i in range(self.n_layers - 1):
            self.conv_layers[i].project()