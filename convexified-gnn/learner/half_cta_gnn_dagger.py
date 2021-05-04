import numpy as np
import os

import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.autograd import Variable

from learner.state_with_delay import MultiAgentStateWithDelay
from learner.replay_buffer import ReplayBuffer
from learner.replay_buffer import Transition
from learner.half_cx_actor import HalfCxActor
from learner.actor import Actor
from learner.cx_graph_filter import tprint

import numpy as np
from numpy import linalg as LA

#
# # TODO: how to deal with bounded/unbounded action spaces?? Should I always assume bounded actions?


class HalfCTADAGGER(object):

    def __init__(self, device, args, k=None):  # , n_s, n_a, k, device, hidden_size=32, gamma=0.99, tau=0.5):
        """
        Initialize the DDPG networks.
        @param device: CUDA device for torch
        @param args: experiment arguments
        """

        n_s = args.getint('n_states')
        n_a = args.getint('n_actions')
        k = k or args.getint('k')
        hidden_size = args.getint('hidden_size')
        n_layers = args.getint('n_layers') or 2
        
        self.n_agents = args.getint('n_agents')
        self.n_states = n_s
        self.n_actions = n_a

        # Device
        self.device = device

        hidden_layers = [hidden_size] * n_layers
        ind_agg = 0  # int(len(hidden_layers) / 2)  # aggregate halfway

        # Define Networks
        self.actor = HalfCxActor(n_s, n_a, hidden_layers, k, ind_agg, device).to(self.device)

        # Define Optimizers
        self.lr = args.getfloat('actor_lr')
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)

        # Prepare an online optimizer
        params = []
        for i in range(self.actor.n_layers - 1):
            params.append(self.actor.conv_layers[i].weight)
            params.append(self.actor.conv_layers[i].bias)
        
        self.cx_actor_optim = Adam(self.param_gen(params), lr=self.lr)

        # Constants
        self.gamma = args.getfloat('gamma')
        self.tau = args.getfloat('tau')
        
        # torch.autograd.set_detect_anomaly(True)

    def select_action(self, state, is_test):
        """
        Evaluate the Actor network over the given state, and with injection of noise.
        @param state: The current state.
        @param graph_shift_op: History of graph shift operators
        @param action_noise: The action noise
        @return:
        """
        # self.actor.eval()  # Switch the actor network to Evaluation Mode.
        mu = self.actor(state.delay_state, state.delay_gso, is_test)  # .to(self.device)

        # mu is (B, 1, nA, N), need (N, nA)
        mu = mu.permute(0, 1, 3, 2)
        mu = mu.view((self.n_agents, self.n_actions))
        
        # self.actor.train()  # Switch back to Train mode.
        mu = mu.data
        return mu

        # return mu.clamp(-1, 1)  # TODO clamp action to what space?
    
    def gradient_step(self, batch):
        """
        Take a gradient step given a batch of sampled transitions.
        @param batch: The batch of training samples.
        @return: The loss function in the network.
        """

        delay_gso_batch = Variable(torch.cat(tuple([s.delay_gso for s in batch.state]))).to(self.device)
        delay_state_batch = Variable(torch.cat(tuple([s.delay_state for s in batch.state]))).to(self.device)
        actor_batch = self.actor(delay_state_batch, delay_gso_batch, False)
        optimal_action_batch = Variable(torch.cat(batch.action)).to(self.device)

        # Optimize Actor
        self.actor_optim.zero_grad()
        # Loss related to sampled Actor Gradient.
        policy_loss = F.mse_loss(actor_batch, optimal_action_batch) #.to(self.device)
        # print(policy_loss)
        # a = list(self.actor.parameters())[0].clone()
        
        self.actor.to(self.device)
        policy_loss.backward() # retain_graph=True
        
        """
        There are scenarios in which SvdBackward returns Nan values for the sake of
        calculating the parameters' gradients. As a countermeasure, we alter the Nan
        values to a relatively small value, such that the parameters will be updated
        and training will be executed without any Runtime errors.
        """
        for name, p in self.actor.named_parameters():
            # print(name)
            if torch.sum(torch.isnan(p.grad)) > 0:
                grad_shape = p.grad.shape
                prod = grad_shape[0]
                for i in range(1, len(grad_shape)):
                    prod *= grad_shape[i]
                grad = p.grad.reshape(prod)
                # for i in range(len(grad)):
                #     if grad[i] != grad[i]:
                #         grad[i] = 0.0001e-04
                # grad[torch.nonzero(grad != grad, as_tuple=True)] = 1e-12
                grad[torch.nonzero(grad != grad)] = 1e-12
                p.grad = grad.reshape(grad_shape).to(self.device)
                # print(p.grad)
            # p.grad[(torch.isnan(p.grad) == True).nonzero()].fill_(0)
            # p.grad[p.grad != p.grad].fill_(0)       
            # print(p.grad)   
          # print(torch.sum(torch.isnan(p.grad)))
          # print(p.grad)
        self.actor.to(self.device)
        self.actor_optim.step()
        
        # print(self.actor.convex_layers[2].weight.grad)
        # b = list(self.actor.parameters())[0].clone()
        # print(torch.equal(a.data, b.data))

        # End Optimize Actor
        self.actor.project_filters()
        
        return policy_loss.item()
    
    def param_gen(self, params):
        for p in params:
            yield p
    
    def initialize_online_learning(self, is_convex):
        """
        """
        # self.actor_per_agent = [self.actor for _ in range(self.n_agents)]
        # self.actor_optim_per_agent = [SGD(filter(lambda p: p.requires_grad, self.actor_per_agent[i].parameters()), lr=self.lr) for i in range(self.n_agents)]
        n_s = self.actor.n_s
        n_a = self.actor.n_a
        hidden_layers = self.actor.hidden_layers
        k = self.actor.k
        ind_agg = self.actor.ind_agg
        self.online_actor = HalfCxActor(n_s, n_a, hidden_layers, k, ind_agg, self.device).to(self.device)
        for i in range(self.actor.n_layers):
            if i == self.actor.n_layers - 1:
                self.online_actor.conv_layers[i] = self.actor.conv_layers[i]
                continue
            if is_convex:
                A = self.actor.conv_layers[i].A_sum
                A -= torch.mean(A, dim=0)
                A /= torch.norm(A) + 1
            else:
                A = self.actor.conv_layers[i].weight.clone()
            
            self.online_actor.conv_layers[i].weight = torch.nn.Parameter(A)
            self.online_actor.conv_layers[i].bias = torch.nn.Parameter(self.actor.conv_layers[i].bias.clone())
            # self.online_actor.convex_layers[i].weight = torch.nn.Parameter(self.actor.convex_layers[i].weight.clone())
            # self.online_actor.convex_layers[i].bias = torch.nn.Parameter(self.actor.convex_layers[i].bias.clone())
        
        self.online_actor_optim = Adam(self.online_actor.parameters(), lr=self.lr)
        # self.actor_per_agent = []
        # for _ in range(self.n_agents):
        #     online_actor = CxActor(n_s, n_a, hidden_layers, k, ind_agg, self.device).to(self.device)
        #     for i in range(self.actor.n_layers):
        #         online_actor.convex_layers[i].weight = torch.nn.Parameter(self.actor.convex_layers[i].weight.clone())
        #         online_actor.convex_layers[i].bias = torch.nn.Parameter(self.actor.convex_layers[i].bias.clone())
        #     self.actor_per_agent.append(online_actor)

        # self.actor_optim_per_agent = [SGD(filter(lambda p: p.requires_grad, self.actor_per_agent[i].parameters()), lr=self.lr) for i in range(self.n_agents)]
    
    def select_action_online(self, state):
        """
        Evaluate the each online Actor network over the given state, and with injection of noise.
        @param state: The current state.
        @param graph_shift_op: History of graph shift operators
        @param action_noise: The action noise
        @return:
        """
        mu = self.online_actor(state.delay_state, state.delay_gso, True)  # .to(self.device)

        # mu is (B, 1, nA, N), need (N, nA)
        mu = mu.permute(0, 1, 3, 2)
        mu = mu.view((self.n_agents, self.n_actions))
        
        # self.actor.train()  # Switch back to Train mode.
        mu = mu.data
        return mu

        # self.actor.eval()  # Switch the actor network to Evaluation Mode.
        # out = []
        # for i in range(self.n_agents):
        #     mu = self.actor_per_agent[i](state.delay_state, state.delay_gso, True)  # .to(self.device)
    
        #     # mu is (B, 1, nA, N), need (N, nA)
        #     mu = mu.permute(0, 1, 3, 2)
        #     mu = mu.view((self.n_agents, self.n_actions))
            
        #     # self.actor.train()  # Switch back to Train mode.
        #     out.append(mu.data[i,:])
        
        # return torch.stack(out)
    
        
    def online_step(self, env, action):
        """
        
        """
        optimal_action = torch.FloatTensor(env.env.controller()).to(self.device)
        action = action.to(self.device)
        # action = torch.FloatTensor(action) #
        
        self.cx_actor_optim.zero_grad()
        policy_loss = F.mse_loss(action, optimal_action)
        policy_loss.requires_grad = True            
        policy_loss.backward()
        self.cx_actor_optim.step()

        self.actor.project_filters()
        # self.online_actor_optim.zero_grad()

        # n_neighbors = np.reshape(np.sum(env.env.adj_mat, axis=1), (env.env.n_agents,1))
        # n_neighbors[n_neighbors == 0] = 1
        # n_neighbors[n_neighbors != 0] += 1           

        # # Storing the current filters
        # w = [[self.actor_per_agent[i].convex_layers[j].weight for j in range(self.actor.n_layers)] for i in range(self.n_agents)]
        # w_local = [[self.actor_per_agent[i].convex_layers[j].weight for j in range(self.actor.n_layers)] for i in range(self.n_agents)]
        # for i in range(self.n_agents):
        #     adj_vec = env.env.adj_mat[i,:]
        #     for j in range(self.actor.n_layers):
        #         for k in range(self.n_agents):
        #             if adj_vec[k] != 0:
        #                 w_local[i][j] = w_local[i][j] + w[k][j]
                
        #         # w_local[i][j] = w_local[i][j] - torch.mean(w_local[i][j], dim=0)
        #         # w_local[i][j] = w_local[i][j] / torch.norm(w_local[i][j])
        #         self.actor_per_agent[i].convex_layers[j].weight = torch.nn.Parameter(w_local[i][j] / n_neighbors[i].data[0])
        
        # for i in range(self.n_agents):
        #     # Optimize Actor       
        #     self.actor_optim_per_agent[i].zero_grad()
        #     policy_loss = F.mse_loss(action, optimal_action)
        #     policy_loss.requires_grad = True            
        #     policy_loss.backward()
        #     self.actor_optim_per_agent[i].step()
        
        # for i in range(self.n_agents):
        #     for j in range(self.actor.n_layers):
        #         weight_norm = self.actor_per_agent[i].convex_layers[j].weight
        #         weight_norm = weight_norm - torch.mean(weight_norm, dim=0)
        #         self.actor_per_agent[i].convex_layers[j].weight = torch.nn.Parameter(weight_norm / torch.norm(weight_norm))
        
        # print(self.actor_per_agent[i].convex_layers[j].weight)
        # exit(0)

    def save_model(self, env_name, suffix="", actor_path=None):
        """
        Save the Actor Model after training is completed.
        @param env_name: The environment name.
        @param suffix: The optional suffix.
        @param actor_path: The path to save the actor.
        @return: None
        """
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/actor_{}_{}".format(env_name, suffix)
        tprint('Saving model to {}'.format(actor_path))
        torch.save(self.actor.state_dict(), actor_path)

    def load_model(self, actor_path, map_location):
        """
        Load Actor Model from given paths.
        @param actor_path: The actor path.
        @return: None
        """
        # print('Loading model from {}'.format(actor_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path, map_location))
            self.actor.to(self.device)


def train_HalfCTADAGGER(env, args, device):
    debug = args.getboolean('debug')
    memory = ReplayBuffer(max_size=args.getint('buffer_size'))
    learner = HalfCTADAGGER(device, args)

    n_a = args.getint('n_actions')
    n_agents = args.getint('n_agents')
    batch_size = args.getint('batch_size')

    n_train_episodes = args.getint('n_train_episodes')
    beta_coeff = args.getfloat('beta_coeff')
    test_interval = args.getint('test_interval')
    n_test_episodes = args.getint('n_test_episodes')

    total_numsteps = 0
    updates = 0
    beta = 1

    stats = {'mean': -1.0 * np.Inf, 'std': 0}

    # for i in range(1):
    for i in range(n_train_episodes):
        # print("episode :" + str(i))
        beta = max(beta * beta_coeff, 0.5)

        state = MultiAgentStateWithDelay(device, args, env.reset(), prev_state=None)

        done = False
        policy_loss_sum = 0
        while not done:

            optimal_action = env.env.controller()
            if np.random.binomial(1, beta) > 0:
                action = optimal_action
            else:
                action = learner.select_action(state, True)
                action = action.cpu().numpy()
            
            next_state, reward, done, _ = env.step(action)

            next_state = MultiAgentStateWithDelay(device, args, next_state, prev_state=state)

            total_numsteps += 1

            # action = torch.Tensor(action)
            notdone = torch.Tensor([not done]).to(device)
            reward = torch.Tensor([reward]).to(device)
                
            # action is (N, nA), need (B, 1, nA, N)
            optimal_action = torch.Tensor(optimal_action).to(device)
            optimal_action = optimal_action.transpose(1, 0)
            optimal_action = optimal_action.reshape((1, 1, n_a, n_agents))

            memory.insert(Transition(state, optimal_action, notdone, next_state, reward))

            state = next_state

        if memory.curr_size > batch_size:
            if i == 0:
                learner.actor.project_filters() 
            for _ in range(args.getint('updates_per_step')):
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))
                policy_loss = learner.gradient_step(batch)
                policy_loss_sum += policy_loss
                updates += 1

        if i % test_interval == 0 and debug:
            # learner.initialize_online_learning()
            test_rewards = []
            for _ in range(n_test_episodes):
                ep_reward = 0
                state = MultiAgentStateWithDelay(device, args, env.reset(), prev_state=None)
                done = False
                while not done:
                    action = learner.select_action(state, True)
                    # print(action)
                    act_norm = action.cpu().numpy()
                    # act_norm -= np.mean(act_norm, axis=0)
                    # act_norm /= LA.norm(act_norm) 
                    # act_norm -= np.mean(act_norm, axis=0)#/ learner.n_actions
                    # print(act_norm)
                    # optimal_action = env.env.controller()
                    # print("----")
                    # print(optimal_action)
                    # exit(0)
                    next_state, reward, done, _ = env.step(act_norm)
                    next_state = MultiAgentStateWithDelay(device, args, next_state, prev_state=state)
                    ep_reward += reward
                    state = next_state
                    # learner.online_step(batch, is_online=True)
                    learner.online_step(env, action)
                    # env.render()
                test_rewards.append(ep_reward)
            
            mean_reward = np.mean(test_rewards)
            if stats['mean'] < mean_reward:
                stats['mean'] = mean_reward
                stats['std'] = np.std(test_rewards)
            
                if debug and args.get('fname'):  # save the best model
                    learner.save_model(args.get('env'), suffix=args.get('fname'))

            if debug:
                statistics = env.get_stats()
                tprint(
                    "Episode: {}, updates: {}, total numsteps: {}, reward: {}, policy loss: {}, vel_diffs: {}, min_dists: {}".format(
                        i, updates,
                        total_numsteps,
                        mean_reward,
                        policy_loss_sum,
                        np.mean(statistics['vel_diffs']),
                        np.mean(statistics['min_dists'])))
        
            
    test_rewards = []
    for _ in range(n_test_episodes):
        ep_reward = 0
        state = MultiAgentStateWithDelay(device, args, env.reset(), prev_state=None)
        done = False
        while not done:
            action = learner.select_action(state, True)
            act_norm = action.cpu().numpy()
            # act_norm -= np.mean(act_norm, axis=0)
            # act_norm /= LA.norm(act_norm) 
            # act_norm /= learner.n_agents
            next_state, reward, done, _ = env.step(act_norm)
            next_state = MultiAgentStateWithDelay(device, args, next_state, prev_state=state)
            ep_reward += reward
            state = next_state
            # learner.gradient_step(batch, is_online=True)
            learner.online_step(env, action)
            # env.render()
        test_rewards.append(ep_reward)

    mean_reward = np.mean(test_rewards)
    stats['mean'] = mean_reward
    stats['std'] = np.std(test_rewards)
    
    statistics = env.get_stats()
    
    stats['vel_diffs'] = statistics['vel_diffs']
    stats['min_dists'] = statistics['min_dists']
    
    if debug and args.get('fname'):  # save the best model
        learner.save_model(args.get('env'), suffix=args.get('fname'))

    env.close()
    return stats
