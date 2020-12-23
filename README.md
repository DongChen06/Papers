# Papers
Papers related to machine learning, deep learning and reinforcement learning

## Contents
* [Reinforcement Learning](#reinforcement-Learning)
  * [Single Agent](#single-agent)
    * [Survey](#survey)
    * [Value-based](#value-based)
    * [Policy-based](#policy-based)
  * [Multi Agent](#multi-agent)
    * [Survey](#survey)
    * [Value-based](#value-based)
    * [Policy-based](#policy-based)
    * [Parameter Sharing](#parameter-sharing)
    * [Graph Convolutional Reinforcement Learning](#graph-convolutional-reinforcement-learning)

* [Autonomous Driving](#autonomous-driving)
  * [Single Agent AD](#single-agent-aD)

# Reinforcement Learning

## Single Agent
### Survey

### Value-based

- **DRQN: Hausknecht, Matthew, and Peter Stone. "Deep recurrent q-learning for partially observable mdps." arXiv preprint arXiv:1507.06527 (2015).**
> To explore well in the particial observed environment, this article investigates the effects of adding recurrency to a Deep Q-Network (DQN) by replacing the first post-convolutional fully-connected layer with a recurrent LSTM. 


### Policy-based


## Multi Agent

### Survey

### Value-based
- **VDN (2017): Sunehag, Peter, et al. "Value-decomposition networks for cooperative multi-agent learning." arXiv preprint arXiv:1706.05296 (2017).**
> novel learned additive value-decomposition approach over individual agents. Implicitly, the value decomposition network aims to learn an optimal linear value decomposition from the team reward signal, by back-propagating the total Q gradient through deep neural networks representing the individual component value functions. 

- **QMIX (2018): Rashid, Tabish, et al. "QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning." arXiv preprint arXiv:1803.11485 (2018).**
> QMIX is a novel value-based method that can train decentralised policies in a centralised end-to-end fashion. QMIX employs a network that estimates joint
action-values as a complex non-linear combination of per-agent values that condition only on local observations.

- **DIAL (2016): Foerster, Jakob, et al. "Learning to communicate with deep multi-agent reinforcement learning." Advances in neural information processing systems. 2016.**
> DQN, parameter sharing: the message is generated together with action-value estimation by each DQN agent, then it is encoded and summed with other input signals at the receiver side. Dial pushes gradients from one agent to another through the communication channel.

- **CommNet (2016): Sukhbaatar, Sainbayar, and Rob Fergus. "Learning multiagent communication with backpropagation." Advances in neural information processing systems. 2016.**
> They propose a model where cooperating agents learn to communicate amongst themselves before taking actions. Each agent is controlled by a deep feed-forward network, which additionally has access to a communication channel carrying a continuous vector. Through this channel, they receive the summed transmissions of other agents.

### Policy-based

- **ConsensusNet (2018): Zhang, Kaiqing, et al. "Fully decentralized multi-agent reinforcement learning with networked agents." arXiv preprint arXiv:1802.08757 (2018).**
> Actor-Critic: the actor step is performed individually by each agent without the need to infer the policies of others. For the critic step, each agent shares its estimate of the value function with its neighbors on the network, so that a consensual estimate is achieved, which is further used in the subsequent actor step.

- **MAAC: Iqbal, Shariq, and Fei Sha. "Actor-attention-critic for multi-agent reinforcement learning." International Conference on Machine Learning. PMLR, 2019.**
> We present an actor-critic algorithm that trains decentralized policies in multiagent settings, using centrally computed critics that share an attention mechanism which selects relevant information for each agent at every timestep. Our attention critic is able to dynamically select which agents to attend to at each time point during training, improving performance in multi-agent domains with complex interactions.

- **NeurComm: Chu, Tianshu, Sandeep Chinchali, and Sachin Katti. "Multi-agent Reinforcement Learning for Networked System Control." arXiv preprint arXiv:2004.01339 (2020).**
> Actor-Critic: introduce a spatial discount factor to stabilize training, especially for non-communicative algorithms. We propose a new neural defferentiable communication protocol to adaptively share information on both system states and agent behaviors. 

### Parameter Sharing

- Gupta, Jayesh K., Maxim Egorov, and Mykel Kochenderfer. **"Cooperative multi-agent control using deep reinforcement learning." International Conference on Autonomous Agents and Multiagent Systems**. Springer, Cham, 2017.

- Lin, Kaixiang, et al. **"Efficient large-scale fleet management via multi-agent deep reinforcement learning."** Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018.



### Graph Convolutional Reinforcement Learning

- Jiang, Jiechuan, et al. **"Graph convolutional reinforcement learning."** arXiv preprint arXiv:1810.09202 (2018).

- Dong, Jiqian, et al. **"A DRL-based Multiagent Cooperative Control Framework for CAV Networks: a Graphic Convolution Q Network."** arXiv preprint arXiv:2010.05437 (2020).


# Autonomous Driving

## Single Agent AD

- **Mavrogiannis, Angelos, Rohan Chandra, and Dinesh Manocha. "B-GAP: Behavior-Guided Action Prediction for Autonomous Navigation." arXiv preprint arXiv:2011.03748 (2020).**
> We use a reinforcement learning-based navigation scheme that uses a proximity graph (graph convolutional network (GCN) of traffic agents and computes a safe trajectory for the ego-vehicle that accounts for aggressive driver maneuvers such as overtaking, over-speeding, weaving, and sudden lane changes. 

- **Bouton, Maxime, et al. "Reinforcement Learning with Iterative Reasoning for Merging in Dense Traffic." arXiv preprint arXiv:2005.11895 (2020).**
> we propose a combination of reinforcement learning and game theory to learn merging behaviors. We design a training curriculum for a reinforcement learning agent using the concept of level-k behavior. This approach exposes the agent to a broad variety of behaviors during training, which promotes learning policies that are robust to model discrepancies.

