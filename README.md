# Papers
Papers related to machine learning, deep learning and reinforcement learning

## Contents
* [Reinforcement Learning](#reinforcement-Learning)
  * [Single Agent](#single-agent)
  * [Multi Agent](#multi-agent)
    * [Collaboration](#collaboration)
    * [Competition](#competition)


# Reinforcement Learning

## Single Agent

## Multi Agent
### Collaboration
- **ConsensusNet: Zhang, Kaiqing, et al. "Fully decentralized multi-agent reinforcement learning with networked agents." arXiv preprint arXiv:1802.08757 (2018).**
> Actor-Critic: the actor step is performed individually by each agent without the need to infer the policies of others. For the critic step, each agent shares its estimate of the value function with its neighbors on the network, so that a consensual estimate is achieved, which is further used in the subsequent actor step.

- **NeurComm: Chu, Tianshu, Sandeep Chinchali, and Sachin Katti. "Multi-agent Reinforcement Learning for Networked System Control." arXiv preprint arXiv:2004.01339 (2020).**
> Actor-Critic: introduce a spatial discount factor to stabilize training, especially for non-communicative algorithms. We propose a new neural defferentiable communication protocol to adaptively share information on both system states and agent behaviors. 

### Competition

