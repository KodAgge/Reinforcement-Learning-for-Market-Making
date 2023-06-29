from cProfile import label
from simple_model_mm_q_learning import linear_decreasing, exponential_decreasing
import matplotlib.pyplot as plt
import numpy as np

"""
Plots the hyperparameter schemes for Q-learning
"""

epsilon_start = 1
epsilon_end = 0.05
epsilon_cutoff = 0.5

alpha_start = 0.5
alpha_end = 1e-5

beta_start = 1
beta_end = 0.05
beta_cutoff = 0.5

n = 4e4

decreasing_factor = (alpha_end / alpha_start) ** (1 / n)

x = []
epsilon = []
beta = []
alpha = []

for i in range(int(n)):
    if i % 1000 == 0:
        x.append(i)
        epsilon.append(
            linear_decreasing(i, n, epsilon_start, epsilon_end, epsilon_cutoff)
        )
        beta.append(linear_decreasing(i, n, beta_start, beta_end, beta_cutoff))
        alpha.append(alpha_start * decreasing_factor**i)

plt.plot(x, epsilon, label="$\epsilon$ - exploration rate", color="purple")
plt.legend()
plt.xlabel("episode")
plt.ylabel("exploration rate")
plt.title("epsilon-greedy policy scheme")
plt.xticks(np.linspace(0, n, 6))
plt.show()

plt.plot(x, beta, label="$\\beta$ - exploration rate", color="purple")
plt.legend()
plt.xlabel("episode")
plt.ylabel("exploration rate")
plt.title("exploring starts scheme")
plt.xticks(np.linspace(0, n, 6))
plt.show()

plt.plot(x, alpha, label="$\\alpha$ - learning rate", color="purple")
plt.legend()
plt.xlabel("episode")
plt.ylabel("learning rate")
plt.title("learning rate scheme")
plt.xticks(np.linspace(0, n, 6))
plt.show()
