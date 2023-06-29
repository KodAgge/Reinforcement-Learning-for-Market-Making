import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize_scalar
from lob_utils.lob_functions import *

"""
================== CREDITS =======================
The code in this file was written by Hanna Hultin.

"""


def estimate_rates_lists(data_dict, event_types):
    """
    Estimate rates of different kinds of events as they are defined in the article
    "algorithmic trading with markov chains"

    Parameters
    ----------
    data_dict : dict containing
        ob : numpy array of shape (m, n, n_levels), last dimension is the number of levels for each limit order book
        level : numpy array of shape (m, n), which level each event is changing
        abs_level : numpy array of shape (m, n), which absolute level each event is changing
        size : numpy array of shape (m, n), integers of size of each event
        event : numpy array of shape (m, n), integers representing which type of event each event is
        time : numpy array of shape (m, n), floats of time between each event
    event_types : dict
        mapping between the values in events and the keys to the types of events

    Returns
    -------
    rates
        dict containing all the computed rates
    """
    data_dict = copy.deepcopy(data_dict)
    for k, v in data_dict.items():
        if k == "ob":
            data_dict[k] = [vv[:-1, ...] for vv in v]
        else:
            data_dict[k] = [vv[1:] for vv in v]

    num_seq = len(data_dict["ob"])

    total_time = np.sum([np.sum(t) for t in data_dict["time"]])
    rates = {}

    lo_sell = [(e == event_types["lo sell"]) for e in data_dict["event"]]
    lo_buy = [(e == event_types["lo buy"]) for e in data_dict["event"]]
    lo = [np.logical_or(lo_sell[j], lo_buy[j]) for j in range(num_seq)]
    mo_bid = [(e == event_types["mo bid"]) for e in data_dict["event"]]
    mo_ask = [(e == event_types["mo ask"]) for e in data_dict["event"]]
    mo = [np.logical_or(mo_bid[j], mo_ask[j]) for j in range(num_seq)]
    cancel_sell = [(e == event_types["cancellation sell"]) for e in data_dict["event"]]
    cancel_buy = [(e == event_types["cancellation buy"]) for e in data_dict["event"]]

    tmp_lo_buy = np.concatenate(
        [data_dict["level"][j][lo_buy[j]] for j in range(num_seq)], axis=0
    )
    rates["lo buy levels"], rates["lo buy"] = np.unique(tmp_lo_buy, return_counts=True)
    tmp_lo_sell = np.concatenate(
        [data_dict["level"][j][lo_sell[j]] for j in range(num_seq)], axis=0
    )
    rates["lo sell levels"], rates["lo sell"] = np.unique(
        tmp_lo_sell, return_counts=True
    )

    rates["lo buy"] = rates["lo buy"] / total_time
    rates["lo sell"] = rates["lo sell"] / total_time
    tmp_lo_size = np.concatenate(
        [data_dict["size"][j][lo[j]] for j in range(num_seq)], axis=0
    )
    rates["lo size"] = np.log(
        np.mean(np.abs(tmp_lo_size)) / (np.mean(np.abs(tmp_lo_size)) - 1)
    )

    rates["mo bid"] = np.sum([np.sum(m) for m in mo_bid]) / total_time
    rates["mo ask"] = np.sum([np.sum(m) for m in mo_ask]) / total_time

    tmp_mo_size = np.concatenate(
        [data_dict["size"][j][mo[j]] for j in range(num_seq)], axis=0
    )
    rates["mo size"] = np.log(
        np.mean(np.abs(tmp_mo_size)) / (np.mean(np.abs(tmp_mo_size)) - 1)
    )

    tmp_cancel_buy = np.concatenate(
        [data_dict["level"][j][cancel_buy[j]] for j in range(num_seq)], axis=0
    )
    rates["cancel buy levels"], rates["cancel buy"] = np.unique(
        tmp_cancel_buy, return_counts=True
    )
    tmp_cancel_sell = np.concatenate(
        [data_dict["level"][j][cancel_sell[j]] for j in range(num_seq)], axis=0
    )
    rates["cancel sell levels"], rates["cancel sell"] = np.unique(
        tmp_cancel_sell, return_counts=True
    )

    # compute maximum likelihood for mo size by considering truncated geometric dist
    s_sum = np.sum(np.abs(tmp_mo_size))
    num_mo = np.sum(np.sum([np.sum(m) for m in mo]))
    v = np.zeros(num_mo)
    i = 0
    for s in range(num_seq):
        for m in range(mo[s].size):
            if mo[s][m]:
                v[i] = (
                    LOB(data_dict["ob"][s][m, :]).ask_volume
                    if mo_ask[s][m]
                    else LOB(data_dict["ob"][s][m, :]).bid_volume
                )
                i += 1
    v = np.abs(v)

    def ll(p):
        return (
            num_mo * np.log(p)
            + (s_sum - num_mo) * np.log(1 - p)
            - np.sum(np.log(1 - (1 - p) ** v), axis=-1).reshape((-1, 1))
        )

    res = minimize_scalar(lambda p: -ll(p), bounds=[0.1, 0.9], method="bounded")
    rates["mo size"] = -np.log(1 - res.x).flat[0]

    avg_buy = (
        np.sum(
            [
                np.sum(
                    np.abs(
                        np.apply_along_axis(
                            lambda x: LOB(x.reshape((2, -1))).q_bid(),
                            -1,
                            data_dict["ob"][j].reshape(
                                (data_dict["ob"][j].shape[0], -1)
                            ),
                        )
                    )
                    * data_dict["time"][j][..., np.newaxis],
                    axis=0,
                )
                for j in range(num_seq)
            ],
            axis=0,
        )
        / total_time
    )
    avg_sell = (
        np.sum(
            [
                np.sum(
                    np.abs(
                        np.apply_along_axis(
                            lambda x: LOB(x.reshape((2, -1))).q_ask(),
                            -1,
                            data_dict["ob"][j].reshape(
                                (data_dict["ob"][j].shape[0], -1)
                            ),
                        )
                    )
                    * data_dict["time"][j][..., np.newaxis],
                    axis=0,
                )
                for j in range(num_seq)
            ],
            axis=0,
        )
        / total_time
    )

    rates["cancel buy"] = rates["cancel buy"] / (
        total_time * avg_buy[rates["cancel buy levels"].astype(int)]
    )
    rates["cancel sell"] = rates["cancel sell"] / (
        total_time * avg_sell[rates["cancel sell levels"].astype(int)]
    )

    return rates


def estimate_frequencies(data_dict, event_types):
    """
    Estimate frequencies of different kinds of events

    Parameters
    ----------
    data_dict : dict containing
        ob : numpy array of shape (m, n, n_levels), last dimension is the number of levels for each limit order book
        level : numpy array of shape (m, n), which level each event is changing
        abs_level : numpy array of shape (m, n), which absolute level each event is changing
        size : numpy array of shape (m, n), integers of size of each event
        event : numpy array of shape (m, n), integers representing which type of event each event is
        time : numpy array of shape (m, n), floats of time between each event
    event_types : dict
        mapping between the values in events and the keys to the types of events

    Returns
    -------
    freq
        dict containing all the computed rates

    """
    data_dict = data_dict.copy()

    freq = {}
    num_events = data_dict["event"].size

    lo_sell = data_dict["event"].squeeze() == event_types["lo sell"]
    lo_buy = data_dict["event"].squeeze() == event_types["lo buy"]
    mo_bid = data_dict["event"].squeeze() == event_types["mo bid"]
    mo_ask = data_dict["event"].squeeze() == event_types["mo ask"]
    cancel_sell = data_dict["event"].squeeze() == event_types["cancellation sell"]
    cancel_buy = data_dict["event"].squeeze() == event_types["cancellation buy"]

    freq["lo buy levels"], freq["lo buy"] = np.unique(
        data_dict["level"][lo_buy], return_counts=True
    )
    freq["lo sell levels"], freq["lo sell"] = np.unique(
        data_dict["level"][lo_sell], return_counts=True
    )

    freq["mo bid"] = np.sum(mo_bid)
    freq["mo ask"] = np.sum(mo_ask)

    freq["cancel buy levels"], freq["cancel buy"] = np.unique(
        data_dict["level"][cancel_buy], return_counts=True
    )
    freq["cancel sell levels"], freq["cancel sell"] = np.unique(
        data_dict["level"][cancel_sell], return_counts=True
    )

    s = 0
    for k, v in freq.items():
        if not any(st in k for st in ["levels", "size"]):
            freq[k] = v / num_events
            s += np.sum(freq[k])

    print(s)
    return freq


def events_to_times(data, times, timefactor=100, end_time=None):
    """
    Given data and time between the data changes,
    compute a new array such that it each new data point is with a fixed time interval instead of for each actual change

    Parameters
    ----------
    data : numpy array of shape (m, n, num_levels)
        last dimension is the number of levels for each limit order book
    times : numpy array of shape (m, n)
        floats of time between each event
    timefactor : int
        the inverse of the fixed time interval
    end_time : time to stop at, if None then stopping at the end of the shortest sequence

    Returns
    -------
    data_times
        numpy array of shape (m, t, num_levels)
        where t is computed using fixed time differences from 0 to the end time

    """
    if end_time is None:
        end_time = np.floor(np.min(np.sum(times, axis=1)))
    num_times = int(end_time * timefactor) + 1
    data_times = np.zeros(((data.shape[0], num_times) + data.shape[2:]))
    time_vals = np.zeros(num_times + 1)
    time_vals[:-1] = np.linspace(0, end_time, num_times)
    time_vals[-1] = np.inf

    for i in range(data.shape[0]):
        j = 1
        total_time = 0
        time_index = 0
        while total_time < end_time:
            if j == times.shape[1]:
                total_time = end_time + 1
            else:
                total_time += times[i, j]
            while total_time >= time_vals[time_index]:
                data_times[i, time_index, ...] = data[i, j - 1, ...]
                time_index += 1
            j += 1
    return data_times


def lineplot(data_list, fun_list, titles, data_labels, timefactor=1, x_label="Events"):
    """
    Visualizes the result of using functions on data using lineplot

    Parameters
    ----------
    data_list : list of numpy arrays
    fun_list : list of functions to run on data
    titles : list of string of same length as fun_list
    data_labels : list of string of same length as data_list
    timefactor : float
        such that the x axis should be divided with
    x_label : string

    Returns
    -------
    figure with subplots for each function
    """
    num_data = len(data_list)
    num_fun = len(fun_list)
    df_list = [[[] for _ in range(num_data)] for _ in range(num_fun)]

    for i in range(num_fun):
        for j in range(num_data):
            df_list[i][j] = pd.DataFrame(fun_list[i](data_list[j]))
            df_list[i][j]["index"] = df_list[i][j].index
            df_list[i][j] = pd.melt(
                df_list[i][j], id_vars="index", value_name="fun", var_name="point"
            )
            df_list[i][j]["timepoint"] = df_list[i][j]["point"] / timefactor

    plt.rcParams.update({"font.size": 20})
    fig = plt.figure(figsize=[12, 10])
    ax = [fig.add_subplot("{}1{}".format(num_fun, i + 1)) for i in range(num_fun)]

    for i in range(num_fun):
        for j in range(num_data):
            sns.lineplot(
                x="timepoint",
                y="fun",
                data=df_list[i][j],
                ci="sd",
                ax=ax[i],
                label=data_labels[j],
            )

    for t in range(len(titles)):
        ax[t].set_title(titles[t])
        ax[t].legend(loc="lower left")
        ax[t].set_xlabel(x_label)
        ax[t].set_ylabel("Price")

    fig.tight_layout()
    return fig


def plot_order_imbalance(
    data_list, time_weights, data_labels, num_bins=100, num_regimes=5, depth=None
):
    """
    Plot histogram and transitions between regimes for order imbalance

    Parameters
    ----------
    data_list : list of numpy arrays
    time_weights : list of numpy arrays for how long time each ob existed
    data_labels : list of string of same length as data_list
    num_bins : int
        number of bins to use for histogram
    num_regimes : int
        number of regimes to use for transition diagram
    depth : int
        what depth to consider for the order imbalance calculation, if None then full depth used

    Returns
    -------
    two figures, first with histogram and second with transition probabilities
    """
    num_data = len(data_list)
    oi = [
        np.apply_along_axis(
            lambda x: LOB(x).order_imbalance(depth),
            -1,
            d[:, :-1, ...].reshape((d.shape[0], d.shape[1] - 1, -1)),
        )
        for d in data_list
    ]

    oi_bins = (2 * np.arange(0, num_bins + 1) / num_bins) - 1

    fig1 = plt.figure(figsize=[12, 8])
    ax1 = fig1.add_subplot(111)
    for i in range(num_data):
        w = time_weights[i][:, 1:]
        _, _, _ = ax1.hist(
            oi[i][np.isfinite(oi[i])],
            bins=oi_bins,
            alpha=0.5,
            label=data_labels[i],
            weights=w[np.isfinite(oi[i])],
            density=True,
        )

    ax1.set_title("Order Imbalance")
    ax1.legend()

    oi_bins = (2 * np.arange(0, num_regimes + 1) / num_regimes) - 1
    oi_bins[-1] += 1e-15
    oi_ind = [np.digitize(o, oi_bins) for o in oi]

    df_trans = [[]] * num_data
    df_oi = [0] * num_data

    for j in range(num_data):
        for i in range(oi_ind[j].shape[0]):
            df_trans[j].append(
                pd.crosstab(
                    pd.Series(oi_ind[j][i, 1:], name="Tomorrow"),
                    pd.Series(oi_ind[j][i, :-1], name="Today"),
                    normalize=1,
                )
            )

        df_oi[j] = pd.concat(df_trans[j]).fillna(0).groupby(level=0).sum()
        df_oi[j] = df_oi[j] / df_oi[j].sum()

    if num_data == 2:
        num_data += 1
        df_oi.append(df_oi[0] - df_oi[1])
        data_labels.append("diff")

    fig2 = plt.figure(figsize=(10 * num_data, 10))
    for d in range(num_data):
        ax2 = fig2.add_subplot(1, num_data, d + 1)
        ax2.matshow(df_oi[d], cmap="seismic")

        for (i, j), z in np.ndenumerate(df_oi[d]):
            ax2.text(
                j,
                i,
                "{:0.3f}".format(z),
                ha="center",
                va="center",
                color="white",
                size=15,
            )

        if data_labels[d] == "diff":
            ax2.set_title("Difference")
        else:
            ax2.set_title("Transitions between states for {}".format(data_labels[d]))

    return fig1, fig2


def market_order_signatures(
    data_dict,
    event_types,
    size_limits=None,
    end_time=10,
    timefactor=100,
    abs_size=False,
):
    """
    Estimate price signatures for market orders

    Parameters
    ----------
    data_dict : dict containing
        ob : numpy array of shape (m, n, n_levels), last dimension is the number of levels for each limit order book
        event : numpy array of shape (m, n), integers representing which type of event each event is
        time : numpy array of shape (m, n), floats of time between each event
        size/abs_size : numpy array of shape (m, n), floats of size of each order
    event_types : dict
        mapping between the values in events and the keys to the types of events
    size_limits : list of two numbers
        [min_size, max_size] such that we only consider market orders fulfilling min_size <= mo size <= max_size
        used to remove outliers which has too much impact, if None then all market orders included
    end_time : int
        how long after to look at for each market order
    timefactor : int
        how many points in time per time unit to use
    abs_size : bool
        whether to use "abs_size" key of data_dict, otherwise "size" is used

    Returns
    -------
    ps
        dict containing price signatures for bid and ask market orders

    """
    if size_limits is None:
        size_limits = [0, np.inf]
    data_dict = data_dict.copy()
    for k, v in data_dict.items():
        if k == "ob":
            data_dict[k] = v[:, :-1, ...]
        else:
            data_dict[k] = v[:, 1:]

    ps = {}

    def mid_func(d):
        return np.apply_along_axis(
            lambda x: LOB(x).mid, -1, d.reshape((d.shape[:-2] + (-1,)))
        )

    time_key = "time" if "abs_time" not in data_dict else "abs_time"

    for m in ["bid", "ask"]:
        mo = data_dict["event"] == event_types["mo {}".format(m)]
        if np.sum(mo) == 0:
            ps[m] = 0
        else:
            sizes = data_dict["size"] if not abs_size else data_dict["abs_size"]
            mo = mo & (sizes >= size_limits[0]) & (sizes <= size_limits[1])

            p0_ind = []
            ob_ind = []
            for i in np.argwhere(mo):
                if np.sum(data_dict[time_key][i[0], i[1] :]) > end_time:
                    p0_ind.append(i)
                    end_index = np.searchsorted(
                        np.cumsum(data_dict[time_key][i[0], i[1] :]), end_time
                    )
                    ob_ind.append(
                        events_to_times(
                            data_dict["ob"][
                                np.newaxis, i[0], i[1] : i[1] + end_index + 1, ...
                            ],
                            data_dict[time_key][
                                np.newaxis, i[0], i[1] : i[1] + end_index + 1
                            ],
                            timefactor=timefactor,
                            end_time=end_time,
                        ).squeeze()
                    )
            if len(p0_ind) > 0:
                ob_ind = np.array(ob_ind)
                p0_ind = np.array(p0_ind)
                p = mid_func(ob_ind)

                p = p[:, np.newaxis, :]
                p0 = p[..., 0]
                ps[m] = price_signature(
                    p, p0, sizes[p0_ind[:, 0], p0_ind[:, 1]].reshape((-1, 1))
                )
            else:
                ps[m] = 0

    return ps
