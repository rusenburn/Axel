====
Axel
====


Introduction
============

Axel includes implementations of modern machine learning algorithms 
which can learn to play gym-like environments 

Requirements
============

* python 3.10.8 environment.
* git

Getting started
===============

* Clone this repo
    ``git clone https://github.com/rusenburn/Axel.git``
* Install python libraries using requirements.txt file
    ``pip install -r requirements.txt``

Content
=======

**P**roximal **P**olicy **O**ptimization (`PPO <https://arxiv.org/abs/1707.06347>`)
----------------------------------------

ppo `have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity.`

**P**olicy **O**ptimization with **P**enalized **P**oint **P**robability **D**istance (`POP3D <https://arxiv.org/abs/1807.00442>`)
-----------------------------------------

POP3D `which is a lower bound to the square of total variance divergence is proposed as another powerful variant of TRPO, Simulation results show that POP3D is highly competitive compared with PPO.`.

**P**hasic **P**olicy **G**radient (`PPG <https://arxiv.org/abs/2009.04416>`)
-----------------------------------------

PPG `a reinforcement learning framework which modifies traditional on-policy actor-critic methods by separating policy and value function training into distinct phases, PPG significantly improves sample efficiency compared to PPO`

TODO
====

* Muzero
* DeepQ Algorithms

Known Issues
============

Due to working alone on this project with a limited resources, It works fine on my gpu
but there were no tests regarding old gpus or high-end gpus, but I will be trying to make it run on cpu
incase there was no gpu, which is not supported atm.