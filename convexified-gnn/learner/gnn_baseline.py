"""
Soruce:
[Tolstaya et al., 2020]
Ekaterina  Tolstaya,  Fernando  Gama,James Paulos, George Pappas, Vijay Kumar, and Alejan-dro Ribeiro.  
Learning decentralized controllers for robotswarms  with  graph  neural  networks.   
In Conference on Robot Learning, pages 671–682, 2020.
"""

import numpy as np


def train_baseline(env, args):
    n_test_episodes = args.getint('n_test_episodes')
    centralized = args.getboolean('centralized')

    stats = {'mean': -1.0 * np.Inf, 'std': 0}

    test_rewards = []
    for _ in range(n_test_episodes):
        ep_reward = 0
        env.reset()
        done = False
        while not done:
            action = env.env.controller(centralized)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            # env.render()
        test_rewards.append(ep_reward)

    mean_reward = np.mean(test_rewards)
    stats['mean'] = mean_reward
    stats['std'] = np.std(test_rewards)
    
    statistics = env.get_stats()
    
    stats['vel_diffs'] = statistics['vel_diffs']
    stats['min_dists'] = statistics['min_dists']

    env.close()
    return stats
