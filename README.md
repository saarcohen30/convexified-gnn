# Convexified Graph Neural Networks for Distributed Control in Robotic Swarms
Code for implementation of convexified graph neural network (Cx-GNN), as well as its extensions, in the context of a swarm of flocking robots. Specifically, our methods are evaluated through the problem of collision-free flocking at the same velocity, a task in which robots aim to align their velocities and regulate their spacing. 
If any part of this code is used, the following paper must be cited: 

Saar Cohen and Noa Agmon. Convexified Graph Neural Networks for Distributed Control in Robotic Swarms. <em>In IJCAI'21: the International Joint Conference on Artificial Intelligence, 2021</em> (to appear).

## Dependencies
Evaluations were conducted using a $12$GB NVIDIA Tesla K80 GPU, implemented in PyTorch v1.7.0, accelerated with Cuda v10.1, and situated in the GymFlock~\cite{tolstaya2020learning} flocking environment.
- Implemented in PyTorch v1.7.0.
- Accelerated with Cuda v10.1.
- Situated in the [GymFlock](https://github.com/katetolstaya/gym-flock) flocking environment.
