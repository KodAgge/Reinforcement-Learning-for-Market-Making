# Reinforcement-Learning-for-Market-Making

This is the GitHub repository for our MSc thesis project _Reinforcement Learning for Market Making_ in financial mathematics at KTH Royal Institute of Technology. The thesis was written during the spring of 2022 in collaboration with Skandinaviska Enskilda Banken and can be found [here](google.com). FIX LINK

## Thesis
Our thesis 

### Abstract
> Market making - the process of simultaneously and continuously providing buy and sell prices in a financial asset - is rather complicated to optimize. Applying reinforcement learning (RL) to infer optimal market making strategies is a relatively uncharted and novel research area. Most published articles in the field are notably opaque concerning most aspects, including precise methods, parameters, and results. This thesis attempts to explore and shed some light on the techniques, problem formulations, algorithms, and hyperparameters used to construct RL-derived strategies for market making. First, a simple probabilistic model of a limit order book is used to compare analytical and RL-derived strategies. Second, a market making agent is trained on a more complex Markov chain model of a limit order book using tabular Q-learning and deep reinforcement learning with double deep Q-learning. Results and strategies are analyzed, compared, and discussed. Finally, we propose some exciting extensions and directions for future work in this research field.



## to do

- [ ] Finish readme
- [ ] Credit Hanna & Hult
- [ ] Add short description of files
- [ ] Add jupyter notebook with examples

## This repository
The code is split into three main categories: 
1. code used to simulate the LOBs and environments
2. code used to train the reinforcement learning agents
3. code used to generate tables and graphs used to evaluate strategies


```
.
├── code
│   ├── environments          <- python code that simulates the environment
|   |   ├── mc_model          <- .py files
│   |   └── simple_model      <- .py files
|   |
│   ├── results               <- plots and q_tables seperated by model
|   |   ├── mc_model          <- .pkl and .png files stored in folders
│   |   └── simple_model      <- .pkl and .png files stored in folders
|   |
│   ├── utils                 <- utils used for the different models and training
|   |   ├── mc_model         
│   |   └── simple_model
|   |
│   └── .py                   <- python files used for Q-learning and evaluating strategies
|
|   
└── README.md

```
--------

## Our contact information
If you want to contact us to discuss our thesis, reinforcement learning, machine learning, algorithmic trading, or anything else, feel free to send us an email or add us on LinkedIn!

| Simon  | [simoncarlsson@outlook.com](mailto:simoncarlsson@outlook.com) | linkedin.com/in/simoncarlsson |
|--------|---------------------------|-------------------------------|
| August | [august.regnell@gmail.com](mailto:august.regnell@gmail.com)  | linkedin.com/in/aregnell      |
