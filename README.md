# Papers
Papers related to machine learning, deep learning and reinforcement learning

# Fresh Papers
- Lee, Hyo-Jun, et al. "MSENet: Marbling score estimation network for automated assessment of Korean beef." Meat Science 188 (2022): 108784.
- Chen, Dian, et al. "Learning by cheating." Conference on Robot Learning. PMLR, 2020.
- Xiao, Xuesu, et al. "Learning Model Predictive Controllers with Real-Time Attention for Real-World Navigation." arXiv preprint arXiv:2209.10780 (2022).
- Evans, Benjamin, et al. "Accelerating Online Reinforcement Learning via Supervisory Safety Systems." arXiv preprint arXiv:2209.11082 (2022).
- Zeng, Yilei, et al. "Human Decision Makings on Curriculum Reinforcement Learning with Difficulty Adjustment." arXiv preprint arXiv:2208.02932 (2022).

## Contents
* [Reinforcement Learning](#reinforcement-Learning)
  * [Survey](#survey)
  * [Value-based](#value-based)
  * [Policy-based](#policy-based)
  * [Model Based](#model-based)
  * [Offline RL](#offline-rl)
  * [Imitation Learning](#imitation-learning)
  * [Inverse Reinforcement Learning](#inverse-reinforcement-learning)
  * [Transfer Learning](#transfer-learning)
  * [Diffusion Models RL](#diffusion-models-rl)
  * [Applications](#applications)
    * [Intelligent Transportation Systems](#intelligent-transportation-systems)
    * [Gaming](#gaming)
    * [Robotics](#robotics)

  
* [Multi Agent Reinforcement Learning](#multi-agent-reinforcement-learning)
  * [Survey](#survey)
  * [Value-based MARL](#value-based-marl)
  * [Policy-based MARL](#policy-based-marl)
  * [Parameter Sharing](#parameter-sharing)
  * [Graph Convolutional Reinforcement Learning](#graph-convolutional-reinforcement-learning)
  * [Offline MARL](#offline-marl)
  * [Multi-agent Imitation Learning](#multi-agent-imitation-learning)
  * [Traffic Applications](#traffic-applications)
    * [Autonomous Driving](#autonomous-driving)
    * [Traffic Signal Control](#traffic-signal-control)

* [Computer Vision](computer-vision)
  * [Image Classification](#image-classification)
  * [Object Detection](#object-detection)
  * [Image Segmentation](#image-segmentation)
  * [GANs](#gans)
  * [Diffusion Models](#diffusion-models)

* [Meta Learning](#meta-Learning)
  * [Meta Learning](#meta-learning)
  * [Meta MARL](#meta-marl)
  * [Offline Meta](#offline-meta)
  * [Traffic Applications](traffic-applications)
  
* [Power System](#power-system)
  * [Voltage and Frequency Control](#voltage-and-frequency-control)
  
* [Smart Agriculture](#smart-agriculture)
  * [Weed Control](#weed-control)
  * [Data Augmentation](#data-augmentation)
  * [Meat Science](#meat-science)
  
* [Robotics](#robotics)
  * [Soft Robots](#soft-robots)
  

* [Tricks](#tricks)
  * [Emsemble](#emsemble)
  * [Curriculum Learning](#curriculum-learning)




# Reinforcement Learning
## Survey

## Value-based

- DRQN: Hausknecht, Matthew, and Peter Stone. **"Deep recurrent q-learning for partially observable mdps."** arXiv preprint arXiv:1507.06527 (2015).
- [Esemble] Lan, Qingfeng, et al. **"Maxmin q-learning: Controlling the estimation bias of q-learning."** arXiv preprint arXiv:2002.06487 (2020).
- [Esemble] Chen, Xinyue, et al. **"Randomized ensembled double q-learning: Learning fast without a model."** arXiv preprint arXiv:2101.05982 (2021).
- [Esemble] Hiraoka, Takuya, et al. **"Dropout Q-Functions for Doubly Efficient Reinforcement Learning."** arXiv preprint arXiv:2110.02034 (2021).

## Policy-based



## Offline RL
- Survey: Levine, Sergey, et al. **"Offline reinforcement learning: Tutorial, review, and perspectives on open problems."** arXiv preprint arXiv:2005.01643 (2020).

- (BCQ): Fujimoto, Scott, David Meger, and Doina Precup. **"Off-policy deep reinforcement learning without exploration."** International Conference on Machine Learning. PMLR, 2019.
- (BEAR) Kumar, Aviral, et al. **"Stabilizing off-policy q-learning via bootstrapping error reduction."** arXiv preprint arXiv:1906.00949 (2019).
- Chen, Lili, et al. **"Decision transformer: Reinforcement learning via sequence modeling."** arXiv preprint arXiv:2106.01345 (2021).
- Janner, Michael, Qiyang Li, and Sergey Levine. **"Reinforcement Learning as One Big Sequence Modeling Problem."** arXiv preprint arXiv:2106.02039 (2021).
- Fujimoto, Scott, and Shixiang Shane Gu. **"A Minimalist Approach to Offline Reinforcement Learning."** arXiv preprint arXiv:2106.06860 (2021).
- Mandlekar, Ajay, et al. **"Iris: Implicit reinforcement without interaction at scale for learning control from offline robot manipulation data."**
  2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020


**Offline-to-Online**
- Nair, Ashvin, et al. **"AWAC: Accelerating Online Reinforcement Learning with Offline Datasets."** (2020).
- Lee, Seunghyun, et al. **"Offline-to-Online Reinforcement Learning via Balanced Replay and Pessimistic Q-Ensemble."** arXiv preprint arXiv:2107.00591 (2021).


## Model Based
- Kurutach, Thanard, et al. **"Model-ensemble trust-region policy optimization."** arXiv preprint arXiv:1802.10592 (2018).
- Matsushima, Tatsuya, et al. **"Deployment-efficient reinforcement learning via model-based offline optimization."** arXiv preprint arXiv:2006.03647 (2020).
- Zhang, Marvin, et al. **"Solar: Deep structured representations for model-based reinforcement learning."** International Conference on Machine Learning. PMLR, 2019.


**Uncertainty Estimate**
- Yu, Tianhe, et al. **"Mopo: Model-based offline policy optimization."** arXiv preprint arXiv:2005.13239 (2020).
- (LOMPO) Rafailov, Rafael, et al. **"Offline reinforcement learning from images with latent space models."** Learning for Dynamics and Control. PMLR, 2021.


## Imitation Learning
- Chen, Dian, et al. "Learning by cheating." Conference on Robot Learning. PMLR, 2020.
- Lynch, Corey, et al. **"Learning latent plans from play."** Conference on Robot Learning. PMLR, 2020.
- (BCQ) Torabi, Faraz, Garrett Warnell, and Peter Stone. **"Behavioral cloning from observation."** arXiv preprint arXiv:1805.01954 (2018).
- (ILPO) Edwards, Ashley, et al. **"Imitating latent policies from observation."** International Conference on Machine Learning. PMLR, 2019.


## Hierarchical Reinforcement Learning
- Nachum, Ofir, et al. **"Data-efficient hierarchical reinforcement learning."** arXiv preprint arXiv:1805.08296 (2018).


## Inverse Reinforcement Learning
- (FORM) Jaegle, Andrew, et al. **"Imitation by Predicting Observations."** International Conference on Machine Learning. PMLR, 2021.


## Transfer Learning
- Cang, Catherine, et al. **"Behavioral Priors and Dynamics Models: Improving Performance and Domain Transfer in Offline RL."** arXiv preprint arXiv:2106.09119 (2021).

## Diffusion Models RL
- Wang, Zhendong, Jonathan J. Hunt, and Mingyuan Zhou. **"Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning."** arXiv preprint arXiv:2208.06193 (2022).
- Janner, Michael, et al. **"Planning with Diffusion for Flexible Behavior Synthesis."** arXiv preprint arXiv:2205.09991 (2022).

## Applications

### Intelligent Transportation Systems
- Mavrogiannis, Angelos, Rohan Chandra, and Dinesh Manocha.  **"B-GAP: Behavior-Guided Action Prediction for Autonomous Navigation."** arXiv preprint arXiv:2011.03748 (2020).

### Gaming
- Zha, Daochen, et al. **"DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning."** arXiv preprint arXiv:2106.06135 (2021).

### Robotics
- Evans, Benjamin, et al. "Accelerating Online Reinforcement Learning via Supervisory Safety Systems." arXiv preprint arXiv:2209.11082 (2022).



# Multi Agent Reinforcement Learning

## Survey MARL
- Da Silva, Felipe Leno, and Anna Helena Reali Costa. **"A survey on transfer learning for multiagent reinforcement learning systems."** Journal of Artificial Intelligence Research 64 (2019): 645-703.
- Wong, Annie, et al. **"Multiagent Deep Reinforcement Learning: Challenges and Directions Towards Human-Like Approaches."** arXiv preprint arXiv:2106.15691 (2021).


## Value-based MARL
- VDN (2017): Sunehag, Peter, et al. **"Value-decomposition networks for cooperative multi-agent learning."** arXiv preprint arXiv:1706.05296 (2017).
- QMIX (2018): Rashid, Tabish, et al. **"QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning."** arXiv preprint arXiv:1803.11485 (2018).
- DIAL (2016): Foerster, Jakob, et al. **"Learning to communicate with deep multi-agent reinforcement learning."** Advances in neural information processing systems. 2016.
- CommNet (2016): Sukhbaatar, Sainbayar, and Rob Fergus. **"Learning multiagent communication with backpropagation."** Advances in neural information processing systems. 2016.
- IAC (2021): Ma, Xiaoteng, et al. "Modeling the Interaction between Agents in Cooperative Multi-Agent Reinforcement Learning." arXiv preprint arXiv:2102.06042 (2021).
 

## Policy-based MARL

- ConsensusNet (2018): Zhang, Kaiqing, et al. **"Fully decentralized multi-agent reinforcement learning with networked agents."** arXiv preprint arXiv:1802.08757 (2018).
- MAAC: Iqbal, Shariq, and Fei Sha. **"Actor-attention-critic for multi-agent reinforcement learning."** International Conference on Machine Learning. PMLR, 2019.
- NeurComm: Chu, Tianshu, Sandeep Chinchali, and Sachin Katti. **"Multi-agent Reinforcement Learning for Networked System Control."** arXiv preprint arXiv:2004.01339 (2020).


## Parameter Sharing
- Gupta, Jayesh K., Maxim Egorov, and Mykel Kochenderfer. **"Cooperative multi-agent control using deep reinforcement learning." International Conference on Autonomous Agents and Multiagent Systems**. Springer, Cham, 2017.
- Lin, Kaixiang, et al. **"Efficient large-scale fleet management via multi-agent deep reinforcement learning."** Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018.


## Graph Convolutional Reinforcement Learning
- Jiang, Jiechuan, et al. **"Graph convolutional reinforcement learning."** arXiv preprint arXiv:1810.09202 (2018).
- Dong, Jiqian, et al. **"A DRL-based Multiagent Cooperative Control Framework for CAV Networks: a Graphic Convolution Q Network."** arXiv preprint arXiv:2010.05437 (2020).


## Offline MARL
- ICQ: Yang, Yiqin, et al. **"Believe What You See: Implicit Constraint Approach for Offline Multi-Agent Reinforcement Learning."** arXiv preprint arXiv:2106.03400 (2021).


## Multi-agent Imitation Learning
- Wang, Hongwei, et al. "Multi-Agent Imitation Learning with Copulas." arXiv preprint arXiv:2107.04750 (2021).


## Traffic Applications

### Autonomous Driving
- self-play: Tang, Yichuan. **"Towards learning multi-agent negotiations via self-play."** Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops. 2019.

### Traffic Signal Control


# Computer Vision
## Image Classification

## Object Detection

## Image Segmentation

## Diffusion Models
- Sohl-Dickstein, Jascha, et al. "Deep unsupervised learning using nonequilibrium thermodynamics." International Conference on Machine Learning. PMLR, 2015.
- Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in Neural Information Processing Systems 33 (2020): 6840-6851.
- Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising diffusion implicit models." arXiv preprint arXiv:2010.02502 (2020).
- Nichol, Alexander Quinn, and Prafulla Dhariwal. "Improved denoising diffusion probabilistic models." International Conference on Machine Learning. PMLR, 2021.
- Dhariwal, Prafulla, and Alexander Nichol. "Diffusion models beat gans on image synthesis." Advances in Neural Information Processing Systems 34 (2021): 8780-8794.
- Ho, Jonathan, et al. "Cascaded Diffusion Models for High Fidelity Image Generation." J. Mach. Learn. Res. 23 (2022): 47-1.



# Meta Learning
## Meta Learning
- (MAML): Finn, Chelsea, Pieter Abbeel, and Sergey Levine. **"Model-agnostic meta-learning for fast adaptation of deep networks."** International Conference on Machine Learning. PMLR, 2017.
- (Reptile): Nichol, Alex, Joshua Achiam, and John Schulman. **"On first-order meta-learning algorithms."** arXiv preprint arXiv:1803.02999 (2018).
- PEARL: Rakelly, Kate, et al. **"Efficient off-policy meta-reinforcement learning via probabilistic context variables." International conference on machine learning.** PMLR, 2019.
- MAML++: Antoniou, Antreas, Harrison Edwards, and Amos Storkey. **"How to train your MAML."** arXiv preprint arXiv:1810.09502 (2018).
- MQL: Fakoor, Rasool, et al. **"Meta-q-learning."** arXiv preprint arXiv:1910.00125 (2019).

## Meta MARL
- Parisotto, Emilio, et al. **"Concurrent meta reinforcement learning."** arXiv preprint arXiv:1903.02710 (2019).
- Chen, Long, et al. **"Multiagent Meta-Reinforcement Learning for Adaptive Multipath Routing Optimization."** IEEE Transactions on Neural Networks and Learning Systems (2021).
- Munir, Md Shirajum, et al. **"Multi-Agent Meta-Reinforcement Learning for Self-Powered and Sustainable Edge Computing Systems."** IEEE Transactions on Network and Service Management (2021).
- Gupta, Abhinav, Angeliki Lazaridou, and Marc Lanctot. **"Meta Learning for Multi-agent Communication."** Learning to Learn-Workshop at ICLR 2021. 2021.


## Offline Meta
- Mitchell, Eric, et al. **"Offline Meta-Reinforcement Learning with Advantage Weighting."** arXiv preprint arXiv:2008.06043 (2020).
- Li, Lanqing, Rui Yang, and Dijun Luo. **"FOCAL: Efficient Fully-Offline Meta-Reinforcement Learning via Distance Metric Learning and Behavior Regularization."** arXiv preprint arXiv:2010.01112 (2020).


## Imitation Learning
- Duan, Yan, et al. **"One-shot imitation learning."** arXiv preprint arXiv:1703.07326 (2017).
- James, Stephen, Michael Bloesch, and Andrew J. Davison. **"Task-embedded control networks for few-shot imitation learning."** Conference on Robot Learning. PMLR, 2018.

## Traffic Applications
- Jaafra, Yesmina, et al. **"Meta-Reinforcement Learning for Adaptive Autonomous Driving."** (2019)
- Ye, Fei, et al. **"Meta Reinforcement Learning-Based Lane Change Strategy for Autonomous Vehicles."** arXiv preprint arXiv:2008.12451 (2020).  
- Hu, Ye, et al. **"Distributed multi-agent meta learning for trajectory design in wireless drone networks."** IEEE Journal on Selected Areas in Communications (2021).


# Power System
## Voltage and Frequency Control
- Wang, Minrui, et al. "Stabilizing Voltage in Power Distribution Networks via Multi-Agent Reinforcement Learning with Transformer." arXiv preprint arXiv:2206.03721 (2022).
- Wang, Jianhong, et al. "Multi-agent reinforcement learning for active voltage control on power distribution networks." Advances in Neural Information Processing Systems 34 (2021): 3271-3284.


## Testbed
- Meinecke, Steffen, et al. "Simbench—a benchmark dataset of electric power systems to compare innovative solutions based on power flow analysis." Energies 13.12 (2020): 3290.


# Smart Agriculture
## Weed Control
- Chen, Dong, et al. **"Performance evaluation of deep transfer learning on multi-class identification of common weed species in cotton production systems."** Computers and Electronics in Agriculture 198 (2022): 107091.
- Dang, Fengying, et al. **"DeepCottonWeeds (DCW): A Novel Benchmark of YOLO Object Detectors for Weed Detection in Cotton Production Systems."** 2022 ASABE Annual International Meeting. American Society of Agricultural and Biological Engineers, 2022.

## Plant Disease
- Paymode, Ananda S., and Vandana B. Malode. **"Transfer Learning for Multi-Crop Leaf Disease Image Classification using Convolutional Neural Network VGG."** Artificial Intelligence in Agriculture 6 (2022): 23-33.

## Data Augmentation
- [Survey] Lu, Yuzhen, et al. **"Generative adversarial networks (GANs) for image augmentation in agriculture: A systematic review."** Computers and Electronics in Agriculture 200 (2022): 107208. 
- [Survey] Xu, Mingle, et al. **"A Comprehensive Survey of Image Augmentation Techniques for Deep Learning."** arXiv preprint arXiv:2205.01491 (2022).


## Meat Science
- Lee, Hyo-Jun, et al. "MSENet: Marbling score estimation network for automated assessment of Korean beef." Meat Science 188 (2022): 108784.


# Robotics
## Soft Robots
- Xiao, Xuesu, et al. "Learning Model Predictive Controllers with Real-Time Attention for Real-World Navigation." arXiv preprint arXiv:2209.10780 (2022).
- Gasoto, Renato, et al. **"A validated physical model for real-time simulation of soft robotic snakes."** 2019 International Conference on Robotics and Automation (ICRA). IEEE, 2019.
- Liu, Xuan, et al. **"Learning to locomote with artificial neural-network and cpg-based control in a soft snake robot."** 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2020.
- Liu, Xuan, Cagdas Onal, and Jie Fu. **"Reinforcement Learning of a CPG-regulated Locomotion Controller for a Soft Snake Robot."** arXiv preprint arXiv:2207.04899 (2022).
- Ji, Guanglin, et al. **"Towards Safe Control of Continuum Manipulator Using Shielded Multiagent Reinforcement Learning."** IEEE Robotics and Automation Letters 6.4 (2021): 7461-7468.
- Li, Guanda, Jun Shintake, and Mitsuhiro Hayashibe. **"Deep Reinforcement Learning Framework for Underwater Locomotion of Soft Robot."** 2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2021.
- Centurelli, Andrea, et al. **"Closed-loop Dynamic Control of a Soft Manipulator using Deep Reinforcement Learning."** IEEE Robotics and Automation Letters 7.2 (2022): 4741-4748.


# Tricks
## Emsemble
- [Esemble] Lan, Qingfeng, et al. **"Maxmin q-learning: Controlling the estimation bias of q-learning."** arXiv preprint arXiv:2002.06487 (2020).
- [Esemble] Chen, Xinyue, et al. **"Randomized ensembled double q-learning: Learning fast without a model."** arXiv preprint arXiv:2101.05982 (2021).
- [Esemble] Hiraoka, Takuya, et al. **"Dropout Q-Functions for Doubly Efficient Reinforcement Learning."** arXiv preprint arXiv:2110.02034 (2021).

## Curriculum Learning
- Chen, Dong, et al. "Deep multi-agent reinforcement learning for highway on-ramp merging in mixed traffic." arXiv preprint arXiv:2105.05701 (2021).
- Liu, Xuan, et al. "Learning to locomote with artificial neural-network and cpg-based control in a soft snake robot." 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2020.
- Zeng, Yilei, et al. "Human Decision Makings on Curriculum Reinforcement Learning with Difficulty Adjustment." arXiv preprint arXiv:2208.02932 (2022).


