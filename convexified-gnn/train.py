from os import path
import configparser
import numpy as np
import random
import gym
import gym_flock
import torch
import sys
import datetime
import time

from learner.gnn_dagger import train_dagger
from learner.agg_gnn_dagger import train_agg_dagger
from learner.gnn_baseline import train_baseline
from learner.cta_gnn_dagger import train_CTADAGGER
from learner.ca_gnn_dagger import train_CADAGGER
from learner.half_cta_gnn_dagger import train_HalfCTADAGGER
from learner.half_ca_gnn_dagger import train_HalfCADAGGER


def tprint(s):
    """ 
    An enhanced print function with time concatenated to the output.
    Source: Convexified Convolutional Neural Networks, by Zhang et al.
    """
    tm_str = time.strftime("%H:%M:%S", time.gmtime(time.time()))
    print(tm_str + ":  " + str(s))
    sys.stdout.flush()


def run_experiment(args):
    # initialize gym env
    env_name = args.get('env')    
    env = gym.make(env_name)

    if isinstance(env.env, gym_flock.envs.flocking.FlockingRelativeEnv):
        env.env.params_from_cfg(args)

    # use seed
    seed = args.getint('seed')
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # initialize params tuple
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    alg = args.get('alg').lower()
    if alg == 'dagger':
        stats = train_dagger(env, args, device)
    elif alg == 'aggdagger':
        stats = train_agg_dagger(env, args, device)
    elif alg == 'baseline':
        stats = train_baseline(env, args)
    elif alg == 'ctadagger':
        stats = train_CTADAGGER(env, args, device)    
    elif alg == 'cadagger':
        stats = train_CADAGGER(env, args, device)    
    elif alg == 'halfctadagger':
        stats = train_HalfCTADAGGER(env, args, device)
    elif alg == 'halfcadagger':
        stats = train_HalfCADAGGER(env, args, device)
    else:
        raise Exception('Invalid algorithm/mode name')
    return stats


def main():
    fname = sys.argv[1]
    config_file = path.join(path.dirname(__file__), fname)
    config = configparser.ConfigParser()
    config.read(config_file)

    printed_header = False

    if config.sections():
        for section_name in config.sections():
            if not printed_header:
                print(config[section_name].get('header'))
                printed_header = True

            stats = run_experiment(config[section_name])
            tprint(section_name + ", " + str(stats['mean']) + ", " + str(stats['std']) + ", vel_diffs(mean): " + str(np.mean(stats['vel_diffs'])) + ", vel_diffs(std): " + str(np.std(stats['vel_diffs'])) + ", min_dists: " + str(np.mean(stats['min_dists'])))
    else:
        val = run_experiment(config[config.default_section])
        print(val)


if __name__ == "__main__":
    main()
