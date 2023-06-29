import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from environments.mc_model.mc_environment import MonteCarloEnv
from collections import defaultdict
import pickle


def heatmap_Q(Q_tab, file_path=None, skip_T=False):
    """
    generates a heatmap based on Q_tab

    Parameters
    ----------
    Q_tab : dictionary
        a dictionary with values for all state-action pairs
    file_path : str
        where the files should be saved
    skip_T : bool
        whether or not the final time step should be included

    Returns
    -------
    None
    """

    optimal_bid = dict()
    optimal_ask = dict()
    optimal_MO = dict()

    plt.figure()
    for state in list(Q_tab.keys()):
        optimal_action = np.unravel_index(Q_tab[state].argmax(), Q_tab[state].shape)
        optimal_bid[state] = optimal_action[0] + 1
        optimal_ask[state] = optimal_action[1] + 1
        optimal_MO[state] = optimal_action[2]

    ser = pd.Series(
        list(optimal_bid.values()), index=pd.MultiIndex.from_tuples(optimal_bid.keys())
    )
    df = ser.unstack().fillna(0)
    if skip_T:
        df = df.iloc[:, :-1]
    fig = sns.heatmap(df)
    fig.set_title("Optimal bid depth")
    fig.set_xlabel("t (grouped)")
    fig.set_ylabel("inventory (grouped)")

    if file_path == None:
        plt.show()
    else:
        plt.savefig(file_path + "opt_bid_heat")
        plt.close()

    plt.figure()
    ser = pd.Series(
        list(optimal_ask.values()), index=pd.MultiIndex.from_tuples(optimal_ask.keys())
    )
    df = ser.unstack().fillna(0)
    if skip_T:
        df = df.iloc[:, :-1]
    fig = sns.heatmap(df)
    fig.set_title("Optimal ask depth")
    fig.set_xlabel("t (grouped)")
    fig.set_ylabel("inventory (grouped)")

    if file_path == None:
        plt.show()
    else:
        plt.savefig(file_path + "opt_ask_heat")
        plt.close()

    plt.figure()
    ser = pd.Series(
        list(optimal_MO.values()), index=pd.MultiIndex.from_tuples(optimal_MO.keys())
    )
    df = ser.unstack().fillna(0)
    if skip_T:
        df = df.iloc[:, :-1]
    fig = sns.heatmap(df)
    fig.set_title("Market order")
    fig.set_xlabel("t (grouped)")
    fig.set_ylabel("inventory (grouped)")

    if file_path == None:
        plt.show()
    else:
        plt.savefig(file_path + "opt_mo_heat")
        plt.close()


def heatmap_Q_std(Q_std, file_path=None):
    """
    Plots a heatmap of the standard deviation of the q-value of the optimal actions

    Parameters
    ----------
    Q_std : defaultdict
        a defaultdict with states as keys and standard deviations as values
    file_path : str
        where the files should be saved

    Returns
    -------
    None
    """

    plt.figure()

    ser = pd.Series(list(Q_std.values()), index=pd.MultiIndex.from_tuples(Q_std.keys()))
    df = ser.unstack().fillna(0)
    fig = sns.heatmap(df)
    fig.set_title("Standard deviation of optimal actions")
    fig.set_xlabel("t (grouped)")
    fig.set_ylabel("inventory (grouped)")

    if file_path == None:
        plt.show()
    else:
        plt.savefig(file_path + "heatmap_of_std")
        plt.close()


def heatmap_Q_n_errors(Q_mean, Q_tables, n_unique=True, file_path=None):
    """
    Plots a heatmap of the difference in optimal actions between runs. Can show number of
    unique actions or number of actions not agreeing with mean optimal.

    Parameters
    ----------
    Q_mean : defaultdict
        a defaultdict with states as keys and mean q-values as values
    Q_tables : list
        a list with defaultdicts with states as keys and q-values as values
    n_unique : bool
        whether or not number of unique actions should be used or not. If False,
        errors compared to mean optimal will be used
    file_path : str
        where the files should be saved

    Returns
    -------
    None
    """

    Q_n_errors = Q_mean

    if n_unique:
        # ----- CALCULATE THE NUMBER OF UNIQUE ACTIONS -----
        title = "Number of unique of optimal actions"
        vmin = 1
        for state in list(Q_mean.keys()):
            opt_action_array = []

            for Q_tab in Q_tables:
                opt_action = np.unravel_index(Q_tab[state].argmax(), Q_tab[state].shape)
                opt_action_array.append(opt_action)

            n_unique_opt_actions = len(set(opt_action_array))

            Q_n_errors[state] = n_unique_opt_actions

    else:
        # ----- CALCULATE THE NUMBER ERRORS COMPARED TO MEAN OPTIMAL -----
        title = "Number of actions not agreeing with mean optimal action"
        vmin = 0
        for state in list(Q_mean.keys()):
            num_errors = 0

            for Q_tab in Q_tables:
                error = np.unravel_index(
                    Q_tab[state].argmax(), Q_tab[state].shape
                ) != np.unravel_index(Q_mean[state].argmax(), Q_mean[state].shape)
                num_errors += error

            Q_n_errors[state] = num_errors

    plt.figure()
    ser = pd.Series(
        list(Q_n_errors.values()), index=pd.MultiIndex.from_tuples(Q_n_errors.keys())
    )
    df = ser.unstack().fillna(0)
    fig = sns.heatmap(df, vmin=vmin, vmax=len(Q_tables))
    fig.set_title(title)
    fig.set_xlabel("t (grouped)")
    fig.set_ylabel("inventory (grouped)")

    if file_path == None:
        plt.show()

    else:
        if n_unique:
            plt.savefig(file_path + "n_unique_opt_actions")
            plt.close()
        else:
            plt.savefig(file_path + "n_errors_compared_to_mean")
            plt.close()


def show_Q(Q_tab, file_path=None):
    """
    plotting the optimal depths from Q_tab

    Parameters
    ----------
    Q_tab : dictionary
        a dictionary with values for all state-action pairs
    file_path : str
        where the files should be saved

    Returns
    -------
    None
    """

    optimal_bid = dict()
    optimal_ask = dict()

    for state in list(Q_tab.keys()):
        optimal_action = np.array(
            np.unravel_index(Q_tab[state].argmax(), Q_tab[state].shape)
        )
        [optimal_bid[state], optimal_ask[state]] = optimal_action[0:2] + 1

    ser = pd.Series(
        list(optimal_bid.values()), index=pd.MultiIndex.from_tuples(optimal_bid.keys())
    )
    df = ser.unstack()
    df = df.T
    df.columns = "q=" + df.columns.map(str)
    df.plot.line(title="Optimal bid depth", style=".-")
    plt.legend(loc="upper right")
    plt.xlabel("t (grouped)")
    plt.ylabel("depth")
    plt.xticks(np.arange(1, df.shape[0] + 1))

    if file_path == None:
        plt.show()
    else:
        plt.savefig(file_path + "opt_bid_strategy")
        plt.close()

    ser = pd.Series(
        list(optimal_ask.values()), index=pd.MultiIndex.from_tuples(optimal_ask.keys())
    )
    df = ser.unstack()
    df = df.T
    df.columns = "q=" + df.columns.map(str)
    df.plot.line(title="Optimal ask depth", style=".-")
    plt.legend(loc="upper right")
    plt.xlabel("t (grouped)")
    plt.ylabel("depth")
    plt.xticks(np.arange(1, df.shape[0] + 1))

    if file_path == None:
        plt.show()
    else:
        plt.savefig(file_path + "opt_ask_strategy")
        plt.close()


def load_Q(filename, default=True):
    """
    loads a Q table from a pkl file

    Parameters
    ----------
    filename : str
        a string for the filename
    default : bool
        if a defaultdictionary or a dictionary should be returned

    Returns
    -------
    Q : dict
        a defaultdictionary/dictionary will all Q tables. the keys are actions and the values are the actual Q tables
    args : dict
        the model parameters
    n : int
        the number of episodes the Q-learning was run for
    rewards : list
        the saved rewards during training
    """

    # Load the file
    file = open("Q_tables/" + filename + ".pkl", "rb")
    Q_raw, args, n, rewards = pickle.load(file)

    # If we don't want a defaultdict, just return a dict
    if not default:
        return Q_raw

    # Find d
    dim = Q_raw[(0, 0)].shape[0]

    # Transform to a default_dict
    Q_loaded = defaultdict(lambda: np.zeros((dim, dim)))
    Q_loaded.update(Q_raw)

    return Q_loaded, args, n, rewards
