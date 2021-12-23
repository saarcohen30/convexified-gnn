# Convexified Graph Neural Networks for Distributed Control in Robotic Swarms
Code for implementation of convexified graph neural network (**<em>Cx-GNN</em>**), as well as its extensions, in the context of a swarm of flocking robots. Specifically, our methods are evaluated through the problem of collision-free flocking at the same velocity, a task in which robots aim to align their velocities and regulate their spacing. 
If any part of this code is used, the following paper must be cited: 

[**Saar Cohen and Noa Agmon. Convexified Graph Neural Networks for Distributed Control in Robotic Swarms. <em>In IJCAI'21: the International Joint Conference on Artificial Intelligence, 2021</em>.**](https://www.ijcai.org/proceedings/2021/0318.pdf)

## Dependencies
Evaluations were conducted using a 12GB NVIDIA Tesla K80 GPU, and:
- Implemented in Python3 with PyTorch v1.7.0.
- Accelerated with Cuda v10.1.
- Situated in the [GymFlock](https://github.com/katetolstaya/gym-flock) flocking environment.

Additional dependencies are:
- [OpenAI Gym](https://github.com/openai/gym)
- [AirSim](https://github.com/openai/gym) (Optional)

## Concept
This repository implements all the convexified GNN architerctures, introduced in the paper cited above. Those architectures appear in the `convexified-gnn/learner` sub-directory, and the relevant files which correspond to our convexified models are listed as follows:
- `cx_graph_filter.py` (`CGraphFilter`) - This code implements a single convexified graph (convolutional) filter, which corresponds to a single layer of a Convexified GNN (**Cx-GNN**).
- `cx_actor.py` (`CxActor`) - An actor implementation, which incorporates `CGraphFilter` so as to perform the convexification.
- `cta_gnn_dagger.py` (`CTADAGGER`) - The implementation of a Convexified Time-Delayed Aggregation GNNs (**<em>CTA-GNNs</em>**).
- `ca_gnn_dagger.py` (`CADAGGER`) - The implementation of a Convexified Aggregation GNNs (**<em>CA-GNNs</em>**).
- `half_cx_actor.py` (`HalfCxActor`) - An actor implementation, which incorporates both a convex GNN and non-convex one.
- `half_cta_gnn_dagger.py` (`HalfCTADAGGER`) - The implementation of a Time-Delayed Aggregation Half-Convex GNNs (**<em>TAHC-GNNs</em>**).
- `half_ca_gnn_dagger.py` (`HalfCADAGGER`) - The implementation of a Aggregation Half-Convex GNNs (**<em>AHC-GNNs</em>**).

## Execution
The `convexified-gnn/cfg` sub-directory comprises of configuration files, which are given to `convexified-gnn/train.py` as an input upon its execution. A possible execution might be:  
`python3 -W ignore convexified_gnn/train.py convexified-gnn/cfg/n-half-cta-k-3.cfg`
where the `-W ignore` flag ignores possible warnings. `convexified-gnn/train.py` then parses the provided configuration using `configparser`, so as to set up the simulative environment of the desired experiment.
