


# Key Ideas of Papers

## Reinforcement Learning algorithms

### Double Q-Learning
Goal: reduces the over-estimation bias.
- (Dueling DQN) Wang, Ziyu, et al. **"Dueling network architectures for deep reinforcement learning."** International conference on machine learning. PMLR, 2016.

- (TD3) Fujimoto, Scott, Herke Hoof, and David Meger. **"Addressing function approximation error in actor-critic methods."** International Conference on Machine Learning. PMLR, 2018.

- (SAC) Haarnoja, Tuomas, et al. **"Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor."** International conference on machine learning. PMLR, 2018.


### Ensemble Models
Goal: avoids the policy from over-fitting to any single model during an episode, leading to more stable learning
- Anschel, Oron, Nir Baram, and Nahum Shimkin. **"Averaged-dqn: Variance reduction and stabilization for deep reinforcement learning."** International conference on machine learning. PMLR, 2017.

- Kurutach, Thanard, et al. **"Model-ensemble trust-region policy optimization."** arXiv preprint arXiv:1802.10592 (2018).


### Self-play
This strategy is a good tool to use in a competitive and cooperative environment.
- Baker, Bowen, et al. **"Emergent tool use from multi-agent autocurricula."** arXiv preprint arXiv:1909.07528 (2019).




# Tricks
## Reinforcement Learning algorithms

### Clipping
Goal: avoid the values out of the range.
- Schulman, John, et al. **"Proximal policy optimization algorithms."** arXiv preprint arXiv:1707.06347 (2017).

