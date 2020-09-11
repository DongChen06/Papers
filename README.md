# Papers
Papers related to machine learning, deep learning and reinforcement learning

## Contents
* [Reinforcement Learning](#reinforcement-Learning)
  * [Single Agent](#single-agent)
  * [Multi Agent](#multi-agent)
    * [Survey](#survey)
    * [Policy-based](#policy-based)
    * [Value-based](#value-based)


# Reinforcement Learning

## Single Agent

## Multi Agent

### Survey


### Policy-based

- **ConsensusNet: Zhang, Kaiqing, et al. "Fully decentralized multi-agent reinforcement learning with networked agents." arXiv preprint arXiv:1802.08757 (2018).**
> Actor-Critic: the actor step is performed individually by each agent without the need to infer the policies of others. For the critic step, each agent shares its estimate of the value function with its neighbors on the network, so that a consensual estimate is achieved, which is further used in the subsequent actor step.

- **NeurComm: Chu, Tianshu, Sandeep Chinchali, and Sachin Katti. "Multi-agent Reinforcement Learning for Networked System Control." arXiv preprint arXiv:2004.01339 (2020).**
> Actor-Critic: introduce a spatial discount factor to stabilize training, especially for non-communicative algorithms. We propose a new neural defferentiable communication protocol to adaptively share information on both system states and agent behaviors. 

### Value-based
- **VDN: Sunehag, Peter, et al. "Value-decomposition networks for cooperative multi-agent learning." arXiv preprint arXiv:1706.05296 (2017).**
> novel learned additive value-decomposition approach over individual agents. Implicitly, the value decomposition network aims to learn an optimal linear value decomposition from the team reward signal, by back-propagating the total Q gradient through deep neural networks representing the individual component value functions. 

- **DIAL: Foerster, Jakob, et al. "Learning to communicate with deep multi-agent reinforcement learning." Advances in neural information processing systems. 2016.**
> DQN, parameter sharing: the message is generated together with action-value estimation by each DQN agent, then it is encoded and summed with other input signals at the receiver side. Dial pushes gradients from one agent to another through the communication channel.
