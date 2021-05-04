import torch
import numpy as np

class MultiAgentState(object):

    def __init__(self, device, args, env_state, prev_state=None, k=None):
        """
        Create the state object that keeps track of the current state and GSO.
        @param device: CUDA device to use with PyTorch
        @param args:
        @param env_state:
        @param prev_state:
        """
        n_states = args.getint('n_states')
        n_agents = args.getint('n_agents')
        k = k or args.getint('k')
        # n_states = args.n_states
        # n_agents = args.n_agents
        # k = args.k

        # split up the state tuple
        state_value, state_network = env_state

        assert state_value.shape == (n_agents, n_states)
        assert state_network.shape == (n_agents, n_agents)
        assert np.sum(np.diag(state_network)) == 0  # assume no self loops

        # reshape values and network to correct shape
        state_value = state_value.transpose(1, 0)
        state_value = state_value.reshape((1, 1, n_states, n_agents))
        state_network = state_network.reshape((1, 1, n_agents, n_agents))

        # move matrices to GPU
        self.values = torch.Tensor(state_value).to(device) # x_t
        self.network = torch.Tensor(state_network).to(device)

        # compute current GSO - powers of the network matrix : I, A_t, A_t^2... A_t^k-1
        self.curr_gso = torch.zeros((1, k, n_agents, n_agents)).to(device)
        self.curr_gso[0, 0, :, :] = torch.eye(n_agents).view((1, 1, n_agents, n_agents)).to(device)  # I
        for k_ind in range(1, k):
            self.curr_gso[0, k_ind, :, :] = torch.matmul(self.network, self.curr_gso[0, k_ind - 1, :, :])


        # x_t, x_t,..., x_t - Used for the sake of reusing the Actor Class.
        self.delay_state = torch.zeros((1, k, n_states, n_agents)).to(device)
        for i in range(k):
            self.delay_state[0, i, :, :] = self.values
        
