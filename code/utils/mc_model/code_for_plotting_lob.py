import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import matplotlib.colors as mcolors

# from lob_utils.lob_functions import *
# from mc_lob_simulation_class import MarkovChainLobModel


def plot_LOB(lob_data):
    ask = lob_data[0, 0]
    spread = -lob_data[1, 0]
    # print(spread)
    buy_volumes = lob_data[1, spread:]  # non-zero buys
    sell_volumes = lob_data[0, spread:]  # non-zero sells
    lob_volumes = np.concatenate((np.flip(buy_volumes), np.zeros((spread - 1,)), sell_volumes))
    lob_volumes = lob_volumes[3:-3]
    prices = np.array(range(ask - len(buy_volumes) - spread + 1, ask + len(sell_volumes)))
    prices = prices[3:-3]
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

    # fig = plt.figure(figsize=(12, 5))
    # ax = fig.add_subplot(1,1,1)
    fig, ((p1, p2), (p3, p4)) = plt.subplots(ncols=2, nrows=2, figsize=(15, 7))

    # Standard
    p1.set_xticks(ticks=prices)
    p1.grid(True, linewidth=1, linestyle='dashed', color="grey")
    p1.set_axisbelow(True)
    p1.bar(x=prices, height=lob_volumes, color=color)
    p1.set_title("Initial LOB")

    # Standard
    p2.set_xticks(ticks=prices)
    p2.grid(True, linewidth=1, linestyle='dashed', color="grey")
    p2.set_axisbelow(True)
    mo = np.zeros((len(prices),))
    mo[7] = 2
    lob_volumes[7] = 0
    mo[8] = lob_volumes[8]
    lob_volumes[8] -= 1
    p2.bar(x=prices, height=mo, color="orange")
    p2.bar(x=prices, height=lob_volumes, color=color)
    p2.set_title("Buy MO arrives")

    p3.set_xticks(ticks=prices)
    p3.grid(True, linewidth=1, linestyle='dashed', color="grey")
    p3.set_axisbelow(True)
    lo = np.zeros((len(prices),))
    lo[5] = -4
    p3.bar(x=prices, height=lo, color="deepskyblue")
    p3.bar(x=prices, height=lob_volumes, color=color)
    p3.set_title("Buy LO arrives")

    lob_volumes[5] = -4


    p4.set_xticks(ticks=prices)
    p4.grid(True, linewidth=1, linestyle='dashed', color="grey")
    p4.set_axisbelow(True)
    lob_volumes[9] -= 1
    canc = np.zeros((len(prices),))
    canc[9] = lob_volumes[9] + 1
    p4.bar(x=prices, height=canc, color="mistyrose")# , edgecolor="red", linewidth=0.5)
    p4.bar(x=prices, height=lob_volumes, color=color)
    p4.set_title("Sell MO cancellation arrives")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)


    # mm_ask = np.zeros((len(prices),))
    # mm_ask[8] = 5
    # mm_bid = np.zeros((len(prices),))
    # mm_bid[7] = -5
    # ax.bar(x=prices, height=mm_ask, color="darkred")
    # ax.bar(x=prices, height=mm_bid, color="darkblue")
    # ax.set_xlabel("price")
    # ax.set_ylabel("volume")


# if __name__ == "__main__":
#     info = {"num_files": 1, "events_per_file": int(2e6), "twap_prob": 0.0, "num_levels": 10,
#             "data_path": "../data/mc-simulated-data/mc-lob"}
#
#     x0 = np.zeros((2, info["num_levels"] + 1), dtype=int)
#     x0[0, 0] = 10000  # initial ask price
#     x0[0, 1:] = 1  # initiated sell volumes
#     x0[1, :] = -1  # initiated spread and buy volumes
#     print(x0)
#     mc_model = MarkovChainLobModel(num_levels=info["num_levels"], ob_start=x0, include_spread_levels=True)
#     print(mc_model.ob.data)
#     # N = int(1e4)
#     # data_dict = mc_model.simulate(N)
#
#     # plt.plot(data_dict["ob"][:, 0, 0])
#
#     # plt.figure()
#     # plt.bar(x=np.array(range(20))-10, height=np.concatenate((data_dict["ob"][-1,1,1:],data_dict["ob"][-1,0,1:])))
#
#     # plot_LOB(data_dict["ob"][938])
#     """plot_LOB(data_dict["ob"][834])
#     print(data_dict["ob"][938])"""
#
#     x = np.array([[9996,    0,    0,    0,    0,   10,    7,    1,    2,    7,    1],
#                   [  -5,    0,    0,    0,    0,   -4,   -6,   -3,   -1,   -1,   -1]])
#     plot_LOB(x)
#
