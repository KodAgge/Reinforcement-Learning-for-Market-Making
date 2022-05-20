import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

sys.path.append("../")
from environments.mc_model.lob_utils.lob_functions import *
from environments.mc_model.mc_lob_simulation_class import MarkovChainLobModel


def plot_LOB(lob_data):
    ask = lob_data[0, 0]
    spread = -lob_data[1, 0]
    # print(spread)
    buy_volumes = lob_data[1, spread:]  # non-zero buys
    sell_volumes = lob_data[0, spread:]  # non-zero sells
    lob_volumes = np.concatenate((np.flip(buy_volumes), np.zeros((spread - 1,)), sell_volumes))
    prices = np.array(range(ask - len(buy_volumes) - spread + 1, ask + len(sell_volumes)))
    # print(prices)
    # print(lob_volumes)
    # print(lob_volumes.shape)
    # print(prices.shape)

    # Coloring the LOB
    color = np.repeat("b", len(lob_volumes))
    color[np.where(lob_volumes > 0)] = "r"

    print("Spread: " + str(spread))
    print("Prices:")
    print(prices)
    print("Volumes:")
    print(lob_volumes)

    plt.figure()
    plt.bar(x=prices, height=lob_volumes, color=color)


if __name__ == "__main__":
    info = {"num_files": 1, "events_per_file": int(2e6), "twap_prob": 0.0, "num_levels": 10,
            "data_path": "../data/mc-simulated-data/mc-lob"}

    x0 = np.zeros((2, info["num_levels"] + 1), dtype=int)
    x0[0, 0] = 100  # initial ask price
    x0[0, 1:] = 1  # initiated sell volumes
    x0[1, :] = -1  # initiated spread and buy volumes
    print(x0)
    mc_model = MarkovChainLobModel(num_levels=info["num_levels"], ob_start=x0, include_spread_levels=True)
    print(mc_model.ob.data)
    N = int(1e4)
    data_dict = mc_model.simulate(N)
    # plt.plot(data_dict["ob"][:, 0, 0])

    # plt.figure()
    # plt.bar(x=np.array(range(20))-10, height=np.concatenate((data_dict["ob"][-1,1,1:],data_dict["ob"][-1,0,1:])))

    plot_LOB(data_dict["ob"][938])
    """plot_LOB(data_dict["ob"][834])
    print(data_dict["ob"][938])"""
