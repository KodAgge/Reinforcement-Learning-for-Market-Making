import functools
import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import time
from datetime import timedelta

# pkg_resources.require("gym==0.23.0")
import pfrl
from pfrl.experiments.evaluator import save_agent
from pfrl.utils.contexts import evaluating
from pfrl import explorers, replay_buffers, agents
import torch as th
from environments.mc_model.mc_environment_deep import *
import logging
from tabulate import tabulate

logger = logging.getLogger(__name__)

CENT_TO_DOLLAR = 100

# ========================================
# ===== SETUP AND TRAINING THE AGENT =====
# ========================================


class DoubleDQNBatch(agents.DoubleDQN):
    """
    Class used when training with batch since get_q needs to be different
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_q(self, obs, max_only=True):
        """
        Compute q values for observation

        Parameters
        ----------
        obs : observation
        max_only: bool of whether to only return maximum q-value

        Returns
        -------
        numpy array of q-values

        """
        with self.eval_mode():
            with th.no_grad(), evaluating(self.model):
                action_value = self._evaluate_model_and_update_recurrent_states(obs)
                if max_only:
                    q = action_value.max.cpu().detach().numpy().astype(float)
                else:
                    q = action_value.q_values.cpu().detach().numpy().astype(float)

            return q


def setup_ddqn_agent(env, info, gpu=-1):
    """
    Creating DDQN PFRL agent

    Parameters
    ----------
    env : environment for agent to act in
    info : dict of info needed
    gpu : int of which gpu to use (-1 for not gpu)
    batch : whether or not batch will be used in training

    Returns
    -------
    double DQN agent

    """
    obs_size = env.observation_space.low.size
    hidden_size = info["hidden_size"]

    # ADVANCED
    q_func = th.nn.Sequential(
        th.nn.Linear(obs_size, hidden_size).double(),
        th.nn.ReLU(),
        th.nn.Linear(hidden_size, hidden_size).double(),
        th.nn.BatchNorm1d(hidden_size).double(),
        th.nn.ReLU(),
        th.nn.Linear(hidden_size, hidden_size).double(),
        th.nn.ReLU(),
        th.nn.Linear(hidden_size, hidden_size).double(),
        th.nn.BatchNorm1d(hidden_size).double(),
        th.nn.ReLU(),
        th.nn.Linear(hidden_size, env.action_space.n).double(),
        pfrl.q_functions.DiscreteActionValueHead(),
    )

    # STANDARD
    # q_func = th.nn.Sequential(
    #     th.nn.Linear(obs_size, hidden_size).double(),
    #     th.nn.ReLU(),
    #     th.nn.Linear(hidden_size, hidden_size).double(),
    #     th.nn.ReLU(),
    #     th.nn.Linear(hidden_size, env.action_space.n).double(),
    #     pfrl.q_functions.DiscreteActionValueHead(),
    # )

    # Use epsilon-greedy for exploration
    explorer = explorers.LinearDecayEpsilonGreedy(
        info["exploration_initial_eps"],
        info["exploration_final_eps"],
        info["exploration_fraction"] * info["n_train"],
        env.action_space.sample,
    )

    opt = th.optim.Adam(
        q_func.parameters(), lr=info["learning_rate_dqn"], weight_decay=0.0
    )
    rbuf = replay_buffers.ReplayBuffer(info["buffer_size"])

    agent = DoubleDQNBatch(
        q_func,
        opt,
        rbuf,
        gpu=gpu,
        gamma=1,
        explorer=explorer,
        replay_start_size=info["replay_start_size"],
        target_update_interval=info["target_update_interval"],
        minibatch_size=info["minibatch_size"],
        update_interval=info["update_interval"],
    )

    return agent


def train_pfrl_agent_batch(
    agent,
    env,
    sample_env,
    info,
    outdir,
    checkpoint_freq=None,
    step_offset=0,
    step_hooks=(),
    suffix="",
    threshold=1e6,
    print_freq=5,
):
    """
    Training PFRL agent

    Parameters
    ----------
    agent : PFRL agent to use in environment
    env : batch environment for agent to act in
    sample_env : non-batch environment for the agent to act in
    outdir : string of directory to save checkpoints in
    checkpoint_freq : integer of checkpoint frequency
    step_offset : integer of iteration to start at
    step_hooks : see PFRL documenttion
    suffix : str used at end of files saved
    threshold : float, interrupt training if q-values larger than this threshold
    print_freq : how often remaining time should be printed

    Returns
    -------
    two arrays with estimated q-values and the corresponding iterations

    """
    num_envs = env.num_envs

    log_interval = info["log_interval"]
    num_estimate = (
        info["num_estimate"] if sample_env.randomize_reset else 1
    )  # if it doesnt randomize it's useless
    steps = info["n_train"]

    assert (sample_env.T / sample_env.dt) * num_envs <= log_interval

    print_frequency = steps / print_freq

    t = step_offset
    if hasattr(agent, "t"):
        agent.t = step_offset

    q_estimate = [
        np.mean(agent.get_q([sample_env.reset() for _ in range(num_estimate)]))
        / info["reward_scale"]
    ]
    t_estimate = [t]
    r_estimate = [0]
    loss = [0]

    t_diff = 0

    # o_0, r_0
    obss = env.reset()

    start_time_sub = time.time()

    episodal_rs_averages = []

    smooth_r = 0

    try:
        while t < steps:
            # a_t
            actions = agent.batch_act(obss)
            # o_{t+1}, r_{t+1}
            obss, rs, dones, _ = env.step(actions)

            episodal_rs_averages.append(np.mean(rs))

            resets = np.zeros(num_envs, dtype=bool)
            agent.batch_observe(obss, rs, dones, resets)

            # Make mask. 0 if done/reset, 1 if pass
            end = np.logical_or(resets, dones)
            not_end = np.logical_not(end)

            for _ in range(num_envs):
                if (t_diff >= log_interval) or (t == steps):
                    # Save estimates
                    q_estimate.append(
                        np.mean(
                            agent.get_q(
                                [sample_env.reset() for _ in range(num_estimate)]
                            )
                        )
                        / info["reward_scale"]
                    )

                    t_estimate.append(t)

                    # Save the reward estimates
                    n_extra_steps = len(episodal_rs_averages) % (
                        sample_env.T / sample_env.dt
                    )
                    avg_episodal_rew = (
                        (sample_env.T / sample_env.dt)
                        * np.mean(
                            episodal_rs_averages[
                                : int((len(episodal_rs_averages) - n_extra_steps))
                            ]
                        )
                        / info["reward_scale"]
                    )
                    r_estimate.append(avg_episodal_rew)  # Only average full episodes

                    if (
                        len(episodal_rs_averages) >= sample_env.T / sample_env.dt
                    ):  # Reset only when atleast a full episode reached
                        episodal_rs_averages = []

                    # Save the losses
                    loss.append(agent.get_statistics()[1][1])

                    smooth_r = 0.97 * smooth_r + 0.03 * avg_episodal_rew
                    logger.debug("- t: %d", t)
                    logger.debug("- q: %d", q_estimate[-1] / CENT_TO_DOLLAR)
                    logger.debug("- r: %d", r_estimate[-1] / CENT_TO_DOLLAR)
                    logger.debug(
                        "- l: %d",
                        loss[-1] / (info["reward_scale"] ** 2) / CENT_TO_DOLLAR,
                    )
                    logger.debug("- smooth r: %d", smooth_r / CENT_TO_DOLLAR)
                    logger.debug("-" * 24)

                    t_diff = 0
                    if q_estimate[-1] > threshold:
                        logger.warning("Q estimates too high, interrupting training.")
                        return t_estimate, q_estimate
                t += 1

                t_diff += 1
                if checkpoint_freq and t % checkpoint_freq == 0:
                    save_agent(agent, t, outdir, logger, suffix="_checkpoint")

                for hook in step_hooks:
                    hook(env, agent, t)

            if t % print_frequency == 0:
                end_time_sub = time.time()

                percentage = "{:.0%}".format(t / steps)

                time_remaining = timedelta(
                    seconds=round(
                        (end_time_sub - start_time_sub)
                        * (steps - t)
                        / print_frequency,
                        2,
                    )
                )

                logger.info(
                    "- Step %d (%s), %s remaining of the run.", t, percentage, time_remaining
                )

                start_time_sub = time.time()

            if t >= steps:
                break
            # Start a new episode
            obss = env.reset(not_end)

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix=f"_except{suffix}")
        raise

    # Save the final model
    save_agent(agent, t, outdir, logger, suffix=f"_finish{suffix}")

    return t_estimate, q_estimate, r_estimate, loss


def train_multiple_agents_batch(info, args, n, outdir, n_runs, gpu=-1):
    """
    Trains several agents and saves them and their estimates during training.
    Agents saved in outdir/model_folder.
    Estimates saved in outdir/estimate_folder.

    Parameters
    ----------
    info : dict
        a dictionary with values used for the DDQN agent
    args : dict
        a dictionary with the parameters of the environment
    n : int
        the number of time steps the agent is trained for
    outdir : str
        the path of the main folder where data will be saved in
    n_runs : int
        how many agents that should be trained
    gpu : int
        the gpu id (negative = cpu, positive = gpu if exists)
    num_envs : int
        how many environments that are used for the batch training

    Returns
    -------
    model_names : list
        a list with strings of which folders the agents are saved
    estimate_names : list
        a list with strings of the names of the file the estimates are saved in

    """

    # The name of the folder the estimates will be saved in
    folder_name = outdir + "estimate_folder"

    # Create the folder if it doesn't exist
    try:
        os.makedirs(folder_name)
    except:
        logger.info("The folder %s already exists.", folder_name)

    save_parameters(args, info, outdir)

    model_names = []
    estimate_names = []

    num_envs = info["num_envs"]

    suffixes = np.arange(n_runs) + 1

    for suffix in suffixes:
        logger.info("Run %d in progress.", suffix)
        start_time_sub = time.time()

        suffix_ = f"_{suffix}"

        # Create environment
        env_function = get_env_function()
        sample_env = env_function(args)
        vec_env = setup_batch_env(args, env_function, num_envs=num_envs, seed=suffix)

        # Setup the agent
        agent = setup_ddqn_agent(vec_env, info, gpu)

        # Where the model will be stored
        model_name = f"{outdir}model_folder/{n}_finish{suffix_}"

        # Train the agent
        t_estimate, q_estimate, r_estimate, loss = train_pfrl_agent_batch(
            agent,
            vec_env,
            sample_env,
            info,
            f"{outdir}model_folder/",
            suffix=suffix_,
        )
        vec_env.close()

        # The name of the estimate file
        estimate_name = f"{folder_name}/estimate{suffix_}"

        file = open(estimate_name, "wb")

        pickle.dump([t_estimate, q_estimate, r_estimate, loss], file)
        file.close()

        # Save the folder and file names
        model_names.append(model_name)
        estimate_names.append(estimate_name)

        # Calculate run time for the single run
        run_time = timedelta(seconds=round(time.time() - start_time_sub, 2))
        logger.info("Finished in %s.", run_time)

        # Calculate the remaining time
        if suffix < len(suffixes):
            remaining_time = timedelta(
                seconds=round(
                    (len(suffixes) - suffix) * (time.time() - start_time_sub), 2
                )
            )
            logger.info("%s remaining of the training.", remaining_time)

    save_file_names(model_names, estimate_names, outdir)

    save_arguments(args, info, outdir)


# ==========================
# ===== HELP FUNCTIONs =====
# ==========================


def load_agent(agent, folder_name):
    """
    Loads a previously saved agent instance

    Parameters
    ----------
    agent : object
        an instance of agent class
    folder_name : str
        the name of the folder where the agent is saved

    Returns
    -------
    agent : object
        the loaded object instance

    """
    agent.load(f"{os.getcwd()}/{folder_name}")

    return agent


def load_estimate(file_name):
    """
    Loads previously estimate of t, q and r during training

    Parameters
    ----------
    file_name : str
        the name of the file where the estimates are saved

    Returns
    -------
    estimates : tuple
        (t_estimate, q_estimate, r_estimate)

    """
    file = open(file_name, "rb")
    return pickle.load(file)


def save_parameters(model_parameters, training_parameters, folder_name):
    """
    Saves the parameters used for the environment and training in a txt file.

    Parameters
    ----------
    model_paramaters : dict
        a dictionary of the parameters used for model/environment
    training_paramaters : dict
        a dictionary of the parameters used for training
    folder_name : str
        where the txt file will be saved

    Returns
    -------
    None

    """
    file_path = f"{folder_name}/parameters.txt"

    with open(file_path, "w") as f:
        f.write("MODEL PARAMETERS\n")
        for key in list(model_parameters.keys()):
            f.write(f"{key} : {model_parameters[key]}\n")

        f.write("\nTRAINING PARAMETERS\n")
        for key in list(training_parameters.keys()):
            f.write(f"{key} : {training_parameters[key]}\n")


def save_file_names(model_names, estimate_names, outdir):
    """
    Saves the given file names into a file.

    Parameters
    ----------
    model_names : list
        a list with the dir of the models
    estimate_names : list
        a list with the dir of the estimates
    outdir : str
        the path of the main folder where data will be saved in

    Returns
    -------
    file_name : str
        the dir of where the file is saved
    """
    # Create the file
    file_name = f"{outdir}file_names.pkl"

    file = open(file_name, "wb")

    # Save the values in the file
    pickle.dump([model_names, estimate_names], file)
    file.close()

    return file_name


def load_file_names(outdir):
    """
    Loads the file names from a file.

    Parameters
    ----------
    file_name : str
        the dir of where the file is saved

    Returns
    -------
    model_names : list
        a list with the dir of the models
    estimate_names : list
        a list with the dir of the estimates
    """
    file_name = f"{outdir}file_names.pkl"

    file = open(file_name, "rb")

    model_names, estimate_names = pickle.load(file)

    return model_names, estimate_names


def save_arguments(args, info, outdir):
    """
    Saves the given file names into a file.

    Parameters
    ----------
    args : dict
        a dict with arguments
    info : list
        a dict with arguments
    outdir : str
        the path of the main folder where data will be saved in

    Returns
    -------
    file_name : str
        the dir of where the file is saved
    """
    # Create the file
    file_name = f"{outdir}arguments.pkl"

    file = open(file_name, "wb")

    # Save the values in the file
    pickle.dump([args, info], file)
    file.close()

    return file_name


def load_arguments(outdir):
    """
    Loads the file names from a file.

    Parameters
    ----------
    file_name : str
        the dir of where the file is saved

    Returns
    -------
    args : dict
        a dict with arguments
    info : list
        a dict with arguments


    """
    file_name = f"{outdir}arguments.pkl"

    file = open(file_name, "rb")

    args, info = pickle.load(file)

    return args, info


# =============================
# ===== EVALUATING AGENTS =====
# =============================


def evaluate_DDQN_batch(outdir, n_test=10, c=1, Q=3, gpu=-1, randomize_start=False):
    """
    Evaluates several previously trained agents. Produces plots and tables that are
    either shown or saved. If saved, they are stored in outdir/image_folder.

    Parameters
    ----------
    model_names : list
        a list with strings of which folders the agents are saved
    estimate_names : list
        a list with strings of the names of the file the estimates are saved in
    args : dict
        a dictionary with the parameters of the environment
    info : dict
        a dictionary with values used for the DDQN agent
    outdir : str
        the path of the main folder where data will be saved in
    n_test : int
        the number of time steps the agents are tested for
    n_states : int or None
        how many random states that should be used (None for neutral state)
    c : int
        the depth of the constant strategy
    gpu : int
        the gpu id (negative = cpu, positive = gpu if exists)
    batch : bool
        whether or not bath_env should be used

    Returns
    -------
    None

    """

    # CREATE FOLDER WHERE IMAGES ARE SAVED
    image_folder_name = f"{outdir}image_folder"
    try:
        os.makedirs(image_folder_name)
    except:
        logger.info("The folder %s already exists.", image_folder_name)

    # FETCH NEEDED INFO
    model_names, estimate_names = load_file_names(outdir)
    args, info = load_arguments(outdir)
    args["randomize_reset"] = randomize_start
    args["phi"] = 0

    # LOAD THE AGENTS
    agents = []
    env_function = get_env_function()
    vec_env = setup_batch_env(args, env_function, num_envs=info["num_envs"])
    for model_name in model_names:
        agents.append(load_agent(setup_ddqn_agent(vec_env, info, gpu), model_name))

    # CREATE GRAPH WITH REWARD AND Q ESTIMATES
    logger.info("Plotting training.")
    plot_training_r_q_l(estimate_names, info, folder_name=image_folder_name)

    # # CREATE HEATMAPS OF STRATEGIES
    logger.info("Plotting strategies.")
    show_strategies_batch(
        agents,
        vec_env,
        args,
        info["n_states"],
        folder_name=image_folder_name,
        Q=Q,
    )

    # EVALUATE THE DIFFERENT RUNS
    if len(model_names) > 1:
        logger.info("Evaluating agents.")
        mean_r = evaluate_multiple_agents_batch(
            args, vec_env, agents, n_test, folder_name=image_folder_name
        )
        best_idx = np.argmax(mean_r)
    else:
        best_idx = 0

    # EVALUTE THE STABILITY OF THE RUNS
    if len(model_names) > 1:
        plot_stability(agents, vec_env, args, info, Q=Q, folder_name=image_folder_name)

    # EVALUATE AGAINST BENCHMARKS
    logger.info("Evaluating benchmarks...")
    evaluate_benchmark_strategies_batch(
        args,
        vec_env,
        agents,
        best_idx,
        n_test,
        folder_name=image_folder_name,
        c=c,
    )

    vec_env.close()

    # VISUALIZE THE STRATEGIES
    logger.info("Visualizing the strategies.")
    visualize_strategies(
        outdir, n_test=n_test, randomize_start=randomize_start, save_mode=True
    )


def sample_strategies(agent, env_function, args, info, n_test):
    """
    Samples an agent for a given amount of episodes.

    Parameters
    ----------
    agent : object
        an agent object
    env_function : function
        a function used to create an environment
    args : dict
        a dict with arguments
    info : list
        a dict with arguments
    n_test : int
        the number of episodes the environment will be simulated for

    Returns
    -------
    Q_t : np.array
        a numpy array with inventory values
    X_t : np.array
        a numpy array with cash values
    V_t : np.array
        a numpy array with value values
    """
    sample_envs = [env_function(args) for _ in range(info["num_envs"])]

    Q_t = []
    X_t = []
    V_t = []

    with agent.eval_mode():
        for n in range(int(n_test / info["num_envs"])):
            q_t = [[] for _ in sample_envs]
            x_t = [[] for _ in sample_envs]
            v_t = [[] for _ in sample_envs]

            # SAMPLE STATES
            statess = [sample_env.reset() for sample_env in sample_envs]

            for i, sample_env in enumerate(sample_envs):
                q_t[i].append(0)
                x_t[i].append(0)
                v_t[i].append(0)

            for t in range(int(args["T"] / args["dt"])):
                actions = agent.batch_act(statess)

                statess = [
                    sample_env.step(action)[0]
                    for action, sample_env in zip(actions, sample_envs)
                ]

                for i, sample_env in enumerate(sample_envs):
                    q = sample_env.Q_t
                    x = sample_env.X_t
                    v = sample_env.X_t + sample_env.H_t
                    q_t[i].append(q)
                    x_t[i].append(x)
                    v_t[i].append(v)

            for q, x, v in zip(q_t, x_t, v_t):
                Q_t.append(q), X_t.append(x), V_t.append(v)

    Q_t = np.array(Q_t)
    X_t = np.array(X_t)
    V_t = np.array(V_t)

    return Q_t, X_t, V_t


def sample_strategies_mean(agents, env_function, args, info, n_test):
    """
    Samples an the mean strategy for a given amount of episodes.

    Parameters
    ----------
    agents : list
        a list of agents
    env_function : function
        a function used to create an environment
    args : dict
        a dict with arguments
    info : list
        a dict with arguments
    n_test : int
        the number of episodes the environment will be simulated for

    Returns
    -------
    Q_t : np.array
        a numpy array with inventory values
    X_t : np.array
        a numpy array with cash values
    V_t : np.array
        a numpy array with value values
    """
    sample_envs = [env_function(args) for _ in range(info["num_envs"])]

    Q_t = []
    X_t = []
    V_t = []

    for n in range(int(n_test / info["num_envs"])):
        q_t = [[] for _ in sample_envs]
        x_t = [[] for _ in sample_envs]
        v_t = [[] for _ in sample_envs]

        # SAMPLE STATES
        statess = [sample_env.reset() for sample_env in sample_envs]

        for i, sample_env in enumerate(sample_envs):
            q_t[i].append(0)
            x_t[i].append(0)
            v_t[i].append(0)

        for t in range(int(args["T"] / args["dt"])):
            actions = mean_opt_action([statess], agents, batch=True)

            statess = [
                sample_env.step(action)[0]
                for action, sample_env in zip(actions, sample_envs)
            ]

            for i, sample_env in enumerate(sample_envs):
                q = sample_env.Q_t
                x = sample_env.X_t
                v = sample_env.X_t + sample_env.H_t
                q_t[i].append(q)
                x_t[i].append(x)
                v_t[i].append(v)

        for q, x, v in zip(q_t, x_t, v_t):
            Q_t.append(q), X_t.append(x), V_t.append(v)

    Q_t = np.array(Q_t)
    X_t = np.array(X_t)
    V_t = np.array(V_t)

    return Q_t, X_t, V_t


def scenario_analysis_plot(env, agents, args, Q=3, save_mode=True, folder_name=None):
    """
    Shows the bid and ask depth for two averages of random states.

    Parameters
    ----------
    agent : object
        an agent object
    env_function : function
        a function used to create an environment
    agents : list
        a list of agents
    args : dict
        a dict with arguments
    Q : int
        2*Q+1 levels will be shown (-Q,Q)
    save_mode : bool
        whether or not the figure should be saved
    folder_name : str
        where the images should be saved

    Returns
    -------
    None
    """
    # Get the states
    states_1 = [[env.reset()[0] for _ in range(env.num_envs)]]
    states_2 = [[env.reset()[0] for _ in range(env.num_envs)]]

    actions_bid_1, actions_ask_1, t_s, q_s = get_opt_action_matrix(
        states_1, Q, args, agents
    )
    actions_bid_2, actions_ask_2, _, _ = get_opt_action_matrix(
        states_2, Q, args, agents
    )

    show_action_grid(
        actions_bid_1,
        actions_ask_1,
        actions_bid_2,
        actions_ask_2,
        t_s,
        q_s,
        save_mode,
        folder_name,
    )


def show_action_grid(bid_1, ask_1, bid_2, ask_2, t, q, save_mode, folder_name):
    """
    Shows the bid and ask depth for two different state in 2x2 grid.

    Parameters
    ----------
    bid_1 : np.array
        the first array with bid depths
    ask_1 : np.array
        the first array with ask depths
    bid_2 : np.array
        the second array with bid depths
    ask_2 : np.array
        the second array with ask depths
    t : np.array
        an array with the time steps used for plotting
    q : np.array
        an array with the inventory levels used for plotting
    save_mode : bool
        whether or not the figure should be saved
    folder_name : str
        where the images should be saved

    Returns
    -------
    None
    """
    # First state
    df_bid_1 = pd.DataFrame(bid_1)
    df_bid_1 = df_bid_1.set_axis(t, axis=1)
    df_bid_1.index = q

    df_ask_1 = pd.DataFrame(ask_1)
    df_ask_1 = df_ask_1.set_axis(t, axis=1)
    df_ask_1.index = q

    # Second state
    df_bid_2 = pd.DataFrame(bid_2)
    df_bid_2 = df_bid_2.set_axis(t, axis=1)
    df_bid_2.index = q

    df_ask_2 = pd.DataFrame(ask_2)
    df_ask_2 = df_ask_2.set_axis(t, axis=1)
    df_ask_2.index = q

    # Subplots
    _, ((ax1, ax2, cb1), (ax3, ax4, cb2)) = plt.subplots(
        2, 3, figsize=(15, 14), gridspec_kw={"width_ratios": [1, 1, 0.08]}
    )

    g1 = sns.heatmap(df_bid_1, vmin=1, vmax=5, cbar=False, ax=ax1)
    g1.set_title("Optimal bid depth first state")
    g1.set_ylabel("inventory")
    g1.set_xlabel("time")

    g2 = sns.heatmap(df_ask_1, vmin=1, vmax=5, cbar_ax=cb1, ax=ax2)
    g2.set_title("Optimal ask depth first state")
    g2.set_ylabel("inventory")
    g2.set_xlabel("time")

    g3 = sns.heatmap(df_bid_2, vmin=1, vmax=5, cbar=False, ax=ax3)
    g3.set_title("Optimal bid depth second state")
    g3.set_ylabel("inventory")
    g3.set_xlabel("time")

    g4 = sns.heatmap(df_ask_2, vmin=1, vmax=5, cbar_ax=cb2, ax=ax4)
    g4.set_title("Optimal ask depth second state")
    g4.set_ylabel("inventory")
    g4.set_xlabel("time")

    if save_mode:
        plt.savefig(f"{folder_name}/scenario")
        plt.close()
    else:
        plt.show()


def show_action_diff(bid_diff, ask_diff, t, q, save_mode, folder_name):
    """
    Shows the difference in aboslute bid and ask depths for two different states.

    Parameters
    ----------
    bid_diff : np.array
        an array with the absolute difference in bid depths
    ask_diff : np.array
        an array with the absolute difference in ask depths
    t : np.array
        an array with the time steps used for plotting
    q : np.array
        an array with the inventory levels used for plotting
    save_mode : bool
        whether or not the figure should be saved
    folder_name : str
        where the images should be saved

    Returns
    -------
    None
    """
    # First state
    df_bid_1 = pd.DataFrame(bid_diff)
    df_bid_1 = df_bid_1.set_axis(t, axis=1)
    df_bid_1.index = q

    df_ask_1 = pd.DataFrame(ask_diff)
    df_ask_1 = df_ask_1.set_axis(t, axis=1)
    df_ask_1.index = q

    # Subplots
    _, ((ax1, ax2, cb1)) = plt.subplots(
        1, 3, figsize=(15, 7), gridspec_kw={"width_ratios": [1, 1, 0.08]}
    )

    g1 = sns.heatmap(df_bid_1, vmin=0, vmax=4, cbar=False, ax=ax1)
    g1.set_title("Optimal bid depth - absolute difference")
    g1.set_ylabel("inventory")
    g1.set_xlabel("time")

    g2 = sns.heatmap(df_ask_1, vmin=0, vmax=4, cbar_ax=cb1, ax=ax2)
    g2.set_title("Optimal ask depth - absolute difference")
    g2.set_ylabel("inventory")
    g2.set_xlabel("time")

    if save_mode:
        plt.savefig(f"{folder_name}/scenario")
        plt.close()
    else:
        plt.show()


def evaluate_benchmark_strategies_batch(
    args,
    env,
    agents,
    best_idx,
    n_test,
    folder_name=None,
    save_mode=True,
    c=1,
):
    """
    Evaluates the best of the previously trained agents and compares against a constant and random strategy.
    Produces plots and tables that are either shown or saved. If saved, they are stored in outdir/image_folder.

    Parameters
    ----------
    args : dict
        a dictionary with the parameters of the environment
    model_name : str
        a string of where the best agent is saved
    n_test : int
        the number of time steps the agents are tested for
    folder_name : str
        where the images should be saved
    save_mode : bool
        whether or not the images should be saved or shown
    c : int
        the depth of the constant strategy

    Returns
    -------
    None

    """
    # Evaluate the best agent
    logger.info("...best agent")
    r_DDQN = compute_reward_agent_batch(
        agents[best_idx], env, n_test, args["T"] / args["dt"]
    ).reshape(-1)

    # Evaluate the mean agent
    logger.info("...mean agent")
    r_DDQN_mean = compute_reward_mean_agent_batch(
        agents, env, num_episodes=n_test, num_steps=args["T"] / args["dt"]
    ).reshape(-1)

    # Evaluate constant strategy
    logger.info("...constant strategy")
    r_constant = compute_reward_benchmark_batch(
        env,
        num_episodes=n_test,
        num_steps=args["T"] / args["dt"],
        c=c,
        strategy="constant",
    ).reshape(-1)

    # Evalute random strategy
    logger.info("...random_strategy")
    r_random = compute_reward_benchmark_batch(
        env, num_episodes=n_test, num_steps=args["T"] / args["dt"], strategy="random"
    ).reshape(-1)

    # Setup for the table
    rewards = [
        r_constant / args["reward_scale"],
        r_random / args["reward_scale"],
        r_DDQN / args["reward_scale"],
        r_DDQN_mean / args["reward_scale"],
    ]

    labels = ["constant (d=" + str(c) + ")", "random", "DDQN (best run)", "DDQN (mean)"]

    headers = [
        "strategy",
        "mean reward",
        "std reward",
        "reward per action",
        "reward per second",
    ]
    rows = []
    for i, label in enumerate(labels):
        rows.append(
            [
                label,
                np.mean(rewards[i]),
                np.std(rewards[i]),
                np.mean(rewards[i]) / (args["T"] / args["dt"]),
                np.mean(rewards[i]) / args["T"],
            ]
        )

    # Save or show the table
    if save_mode:
        with open(f"{folder_name}/table_benchmarking", "w") as f:
            f.write(tabulate(rows, headers=headers))
    else:
        logger.info("Results:")
        print(tabulate(rows, headers=headers))

    # Save or show the boxplot
    plt.figure(figsize=(12, 5))
    plt.boxplot(rewards, labels=labels)
    plt.title("Comparison of different strategies")
    plt.ylabel("reward")

    if save_mode:
        plt.savefig(f"{folder_name}/box_plot_benchmarking")
        plt.close()
    else:
        plt.show()


def compute_reward_benchmark_batch(
    env, c=1, num_episodes=int(1e3), num_steps=1, strategy="random"
):
    """
    Compute rewards when following the policy given by predicting with the model

    Parameters
    ----------
    agent : object having the function batch_act
    env : environment to use to take next step
    num_episodes : integer of number of episodes to run

    Returns
    -------
    cost_end : numpy array of total cost for each episode
    reward_end : numpy array of total rewards for each episode
    """
    reward_end = np.empty((num_episodes, 1))

    num_envs = env.num_envs

    env.reset()

    for i in range(int(num_episodes / num_envs)):
        r_all = np.zeros(num_envs)

        for _ in range(int(num_steps)):
            if strategy == "random":
                actions = [env.action_space.sample() for _ in range(num_envs)]
            elif strategy == "constant":
                actions = [6 * (c - 1) for _ in range(num_envs)]  # HARD CODED
            _, r, _, _ = env.step(actions)
            r_all += r

        for j, r in enumerate(r_all):
            reward_end[num_envs * i + j] = r_all[j]

        env.reset()

    return reward_end / CENT_TO_DOLLAR


def evaluate_multiple_agents_batch(
    args, env, agents, n_test, n_states=10000, folder_name=None, save_mode=True
):
    """
    Evaluates the previously trained agents and compares the against eachother.
    Produces plots and tables that are either shown or saved. If saved, they are stored in outdir/image_folder.

    Parameters
    ----------
    args : dict
        a dictionary with the parameters of the environment
    info : dict
        a dictionary with values used for the DDQN agent
    n_test : int
        the number of time steps the agents are tested for
    folder_name : str
        where the images should be saved
    save_mode : bool
        whether or not the images should be saved or shown
    gpu : int
        the gpu id (negative = cpu, positive = gpu if exists)

    Returns
    -------
    rewards : array
        an array of the average rewards of the different agents
    """
    rewards = []
    q_values = []

    n_states = n_states if args["randomize_reset"] else 1

    statess = [env.reset() for _ in range(n_states)]

    # Get the rewards of all the agents
    for i, agent in enumerate(agents):
        with agent.eval_mode():
            rewards.append(
                compute_reward_agent_batch(
                    agent, env, n_test, args["T"] / args["dt"]
                ).reshape(-1)
                / args["reward_scale"]
            )

            q_value = 0
            for states in statess:
                q_value += np.mean(agent.get_q(states))

            q_values.append(
                q_value / (n_states * args["reward_scale"]) / CENT_TO_DOLLAR
            )

    # Setup the table
    labels = [f"run {i+1}" for i in range(len(agents))]

    headers = [
        "run",
        "mean reward",
        "std reward",
        "reward per action",
        "reward per second",
        "v*(0,0)",
    ]
    rows = []

    for i, label in enumerate(labels):
        rows.append(
            [
                label,
                np.mean(rewards[i]),
                np.std(rewards[i]),
                np.mean(rewards[i]) / (args["T"] / args["dt"]),
                np.mean(rewards[i]) / args["T"],
                q_values[i],
            ]
        )

    # Save or show the table
    if save_mode:
        with open(f"{folder_name}/table_different_runs", "w") as f:
            f.write(tabulate(rows, headers=headers))
    else:
        logger.info("Results:")
        print(tabulate(rows, headers=headers))

    # Save or show the boxplot
    plt.figure(figsize=(12, 5))
    plt.boxplot(rewards, labels=labels)
    plt.title("Comparison of different Q-learning runs")
    plt.ylabel("reward")

    if save_mode:
        plt.savefig(f"{folder_name}/box_plot_different_runs")
        plt.close()
    else:
        plt.show()

    return np.mean(rewards, axis=1)


def a_from_agent(agent, env):
    """
    Find the policy given by following predictions made by the model.

    Parameters
    ----------
    agent : object having the function act
    env : environment to use to take next step

    Returns
    -------
    a : numpy array of the actions
    """
    with agent.eval_mode():
        a = []
        obs = env.reset()
        for t in range(env.num_action_times):
            action = agent.act(obs)
            action_tuple = (
                int(action / env.max_quote_depth) + 1,
                action % env.max_quote_depth + 1,
            )
            a.append(action_tuple)
            obs, _, done, _ = env.step(action)
            if done:
                break
        return a


def compute_reward_agent_batch(agent, env, num_episodes=int(1e3), num_steps=1):
    """
    Compute rewards when following the policy given by predicting with the model

    Parameters
    ----------
    agent : object having the function batch_act
    env : environment to use to take next step
    num_episodes : integer of number of episodes to run

    Returns
    -------
    cost_end : numpy array of total cost for each episode
    reward_end : numpy array of total rewards for each episode
    """
    reward_end = np.empty((num_episodes, 1))

    num_envs = env.num_envs

    current_state = env.reset()

    with agent.eval_mode():
        for i in range(int(num_episodes / num_envs)):
            r_all = np.zeros(num_envs)

            for _ in range(int(num_steps)):
                actions = agent.batch_act(current_state)
                current_state, r, _, _ = env.step(actions)
                r_all += r

            for j, r in enumerate(r_all):
                reward_end[num_envs * i + j] = r_all[j]

            current_state = env.reset()

    return reward_end / CENT_TO_DOLLAR


def compute_reward_mean_agent_batch(agents, env, num_episodes=int(1e3), num_steps=1):
    """
    Compute rewards when following the policy given by predicting with the model

    Parameters
    ----------
    agent : object having the function batch_act
    env : environment to use to take next step
    num_episodes : integer of number of episodes to run

    Returns
    -------
    cost_end : numpy array of total cost for each episode
    reward_end : numpy array of total rewards for each episode
    """
    reward_end = np.empty((num_episodes, 1))

    num_envs = env.num_envs

    current_state = env.reset()

    for i in range(int(num_episodes / num_envs)):
        r_all = np.zeros(num_envs)

        for _ in range(int(num_steps)):
            actions = mean_opt_action([current_state], agents, batch=True)
            current_state, r, _, _ = env.step(actions)
            r_all += r

        for j, r in enumerate(r_all):
            reward_end[num_envs * i + j] = r_all[j]

        current_state = env.reset()

    return reward_end / CENT_TO_DOLLAR


def mean_opt_action(statess, agents, batch=False):
    """
    Either returns the average optimal action over a set of states or
    the optimal action for every state of a set of states.

    Parameters
    ----------
    statess : list
        a list of lists of states
    agents : list
        a list of agents
    batch : bool
        whether or not the average optimal action or all optimal actions should be returned

    Returns
    -------
    action : int OR list
        the average optimal action or a list of optimal actions
    """
    q_values = 0
    if not batch:
        for agent in agents:
            with agent.eval_mode():
                for states in statess:
                    q_valuess = agent.get_q(states, max_only=False)
                    q_values += np.sum(q_valuess, axis=0)

        action = np.argmax(q_values)

    else:
        for agent in agents:
            with agent.eval_mode():
                for states in statess:
                    q_values += agent.get_q(states, max_only=False)

        action = np.argmax(q_values, axis=1)

    return action


# ====================
# ===== PLOTTING =====
# ====================


def get_opt_action_matrix(statess, Q, args, agents):
    """
    Returns the mean strategy.

    Parameters
    ----------
    statess : list
        a list of lists of states
    Q : int
        the inventory levels the strategy will be shown for (-Q,Q)
    args : dict
        a dictionary of arguments used for the environment
    agents : list
        a list of agents

    Returns
    -------
    actions_bid : np.array
        an array with the optimal bid depths
    actions_ask : np.array
        an array with the optimal ask depths
    t_s : np.array
        an array with the time steps used for plotting
    q_s : np.array
        an array with the inventory levels used for plotting
    """
    actions_bid = np.zeros((2 * Q + 1, int(args["T"] / args["dt"] + 1)))
    actions_ask = np.zeros((2 * Q + 1, int(args["T"] / args["dt"] + 1)))

    q_s = np.arange(-Q, Q + 1)
    t_s = np.arange(int(args["T"] / args["dt"] + 1)) * args["dt"]

    # For every combination of inventory level and time, find the best
    # action based on the given states
    for i, q in enumerate(q_s):
        for j, t in enumerate(t_s):
            for states in statess:
                for state in states:
                    # Normalize
                    state[0] = q / 100
                    state[1] = t / args["T"]

            action = mean_opt_action(statess, agents)

            # Translate to depths
            actions_bid[i, j], actions_ask[i, j] = (
                int(action / 5 + 1),
                action % 5 + 1,
            )  # HARD CODED; FIX THIS

    return actions_bid, actions_ask, t_s, q_s


def show_opt_action(
    actions_bid, actions_ask, t_s, q_s, folder_name, suffix=None, save_mode=True
):
    """
    Plots heatmaps of optimal bid and ask depths.

    Parameters
    ----------
    actions_bid : np.array
        an array with the optimal bid depths
    actions_ask : np.array
        an array with the optimal ask depths
    t_s : np.array
        an array with the time steps used for plotting
    q_s : np.array
        an array with the inventory levels used for plotting
    folder_name : str
        where the figures will be saved
    suffix : str
        string put in the title
    save_mode : bool
        whether or not the figures should be saved or displayed

    Returns
    -------
    None
    """
    # Create the bid plot
    df_bid = pd.DataFrame(actions_bid)
    df_bid = df_bid.set_axis(t_s, axis=1)
    df_bid.index = q_s

    plt.figure()

    fig = sns.heatmap(df_bid, vmin=1, vmax=5)  # HARD CODED
    fig.set_title(f"Optimal bid depth ({suffix})")
    fig.set_xlabel("t")
    fig.set_ylabel("inventory")

    if save_mode:
        plt.savefig(f"{folder_name}/bid_heat_{suffix}")
        plt.close()
    else:
        plt.show()

    # Create the ask plot
    df_ask = pd.DataFrame(actions_ask)
    df_ask = df_ask.set_axis(t_s, axis=1)
    df_ask.index = q_s

    plt.figure()

    fig = sns.heatmap(df_ask, vmin=1, vmax=5)  # HARD CODED
    fig.set_title(f"Optimal ask depth ({suffix})")
    fig.set_xlabel("t")
    fig.set_ylabel("inventory")

    if save_mode:
        plt.savefig(f"{folder_name}/ask_heat_{suffix}")
        plt.close()
    else:
        plt.show()


def show_strategies_batch(
    agents, env, args, n_states=None, Q=3, folder_name=None, save_mode=True
):
    """
    Shows the optimal depths in heatmap form for different states

    Parameters
    ----------
    agents : list
        a list of agents
    env : object
        an instance of the environment class
    args : dict
        a dict containing parameters
    n_states : int or None
        how many random states that should be used (None for neutral state)
    Q : int
        -Q -> Q | for which inventory levels the depths should be shown
    folder_name : str
        where the images should be saved
    save_mode : bool
        whether or not the images should be saved or shown

    Returns
    -------
    None
    """
    n_states = max(10, n_states)

    # Get the states
    if not args["randomize_reset"]:
        statess = [env.reset()]
        suffix = "neutral"
    else:
        statess = [env.reset() for _ in range(n_states)]
        suffix = f"randomized_{n_states}"

    actions_bid, actions_ask, t_s, q_s = get_opt_action_matrix(statess, Q, args, agents)

    show_opt_action(
        actions_bid,
        actions_ask,
        t_s,
        q_s,
        folder_name,
        suffix=suffix,
        save_mode=save_mode,
    )


def plot_training_r_q_l(estimate_names, info, folder_name=None, save_mode=True):
    """
    Plots the rewards, losses and q-estimates calculated during training.
    A CI of one std is also shown for all the different runs.

    Parameters
    ----------
    estimate_names : list
        a list of strings of where the estimates are saved
    older_name : str
        a string of where the plot will be saved
    save_mode : bool
        whether or not the images should be saved or shown

    Returns
    -------
    None
    """
    q_estimates = []
    r_estimates = []
    l_estimates = []

    for estimate_name in estimate_names:
        t, q, r, l = load_estimate(estimate_name)

        q_estimates.append(q)
        r_estimates.append(r)
        l_estimates.append(l)

    r_matrix = np.array(r_estimates) / CENT_TO_DOLLAR
    q_matrix = np.array(q_estimates) / CENT_TO_DOLLAR
    l_matrix = np.array(l_estimates) / (info["reward_scale"] ** 2) / CENT_TO_DOLLAR

    if True:
        q_matrix[:, 0] = q_matrix[:, 1]  # DUE TO BUG IN CODE
        r_matrix[:, 0] = r_matrix[:, 1]  # DUE TO BUG IN CODE

    r_mean = np.mean(r_matrix, axis=0)
    r_std = np.std(r_matrix, axis=0)

    q_mean = np.mean(q_matrix, axis=0)
    q_std = np.std(q_matrix, axis=0)

    l_mean = np.mean(l_matrix, axis=0)
    l_std = np.std(l_matrix, axis=0)

    reward_area = np.array([r_std, -r_std]) + r_mean
    Q_zero_area = np.array([q_std, -q_std]) + q_mean
    loss_area = np.array([l_std, -l_std]) + l_mean

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))

    # Plot the rewards
    ax1.fill_between(
        t,
        reward_area[0, :],
        reward_area[1, :],
        alpha=0.3,
        color="purple",
        label="±$\sigma$",
    )
    ax1.plot(t, r_mean, linewidth=0.4, color="purple", label="mean reward")
    ax1.set_xlabel("step")
    ax1.set_ylabel("reward")
    ax1.set_title("average reward during training")

    # Plot the Q-values
    ax2.fill_between(
        t,
        Q_zero_area[0, :],
        Q_zero_area[1, :],
        alpha=0.3,
        color="purple",
        label="±$\sigma$",
    )
    ax2.plot(t, q_mean, linewidth=0.4, color="purple", label="mean q-estimate")
    ax2.set_xlabel("step")
    ax2.set_ylabel("q-estimate")
    ax2.set_title("average q-estimate during training")

    # Plot the losses
    ax3.fill_between(
        t,
        loss_area[0, :],
        loss_area[1, :],
        alpha=0.3,
        color="purple",
        label="±$\sigma$",
    )
    ax3.plot(t, l_mean, linewidth=0.4, color="purple", label="mean loss")
    ax3.set_xlabel("step")
    ax3.set_ylabel("loss")
    ax3.set_title("average loss during training")

    ax1.legend()
    ax2.legend()
    ax3.legend()

    if save_mode:
        plt.savefig(f"{folder_name}/training_graph")
        plt.close()
    else:
        plt.show()


def visualize_strategies(
    outdir, n_test=10, gpu=-1, save_mode=True, randomize_start=False
):
    """
    Evaluates several previously trained agents. Produces plots and tables that are
    either shown or saved. If saved, they are stored in outdir/image_folder.

    Parameters
    ----------
    outdir : str
        the path of the main folder where data will be saved in
    n_test : int
        the number of time steps the agents are tested for

    Returns
    -------
    None
    """

    # CREATE FOLDER WHERE IMAGES ARE SAVED
    image_folder_name = f"{outdir}image_folder"
    try:
        os.makedirs(image_folder_name)
    except:
        logger.info("The folder %s already exists.", image_folder_name)

    # FETCH NEEDED INFO
    model_names, _ = load_file_names(outdir)
    args, info = load_arguments(outdir)

    args["randomize_reset"] = randomize_start
    args["phi"] = 0

    # LOAD THE AGENTS
    agents = []
    env_function = get_env_function()
    vec_env = setup_batch_env(args, env_function, num_envs=info["num_envs"])
    for model_name in model_names:
        agents.append(load_agent(setup_ddqn_agent(vec_env, info, gpu), model_name))

    # PLOT THE MEAN STRATEGY WITH CI
    Qs, Xs, Vs = sample_strategies_mean(agents, env_function, args, info, n_test)

    _, (q_axis, x_axis, v_axis) = plt.subplots(1, 3, figsize=(21, 7))

    q_axis.set_title("The inventory process - $Q_t$")
    x_axis.set_title("The cash process - $X_t$")
    v_axis.set_title("The value process - $V_t$")

    q_std = np.std(Qs, axis=0)
    q_mean = np.mean(Qs, axis=0)
    x_mean = np.mean(Xs, axis=0) / CENT_TO_DOLLAR
    x_std = np.std(Xs, axis=0) / CENT_TO_DOLLAR
    v_mean = np.mean(Vs, axis=0) / CENT_TO_DOLLAR
    v_std = np.std(Vs, axis=0) / CENT_TO_DOLLAR

    q_axis.plot(q_mean, color="purple", label="mean inventory")
    q_axis.fill_between(
        list(range(len(q_mean))),
        q_mean - q_std,
        q_mean + q_std,
        alpha=0.3,
        color="purple",
        label="±$\sigma$",
    )
    q_axis.set_xlabel("t")
    q_axis.set_ylabel("$Q_t$")
    q_axis.get_yaxis().get_major_formatter().set_useOffset(False)

    x_axis.plot(x_mean, color="purple", label="mean cash")
    x_axis.fill_between(
        list(range(len(x_mean))),
        x_mean - x_std,
        x_mean + x_std,
        alpha=0.3,
        color="purple",
        label="±$\sigma$",
    )
    x_axis.set_xlabel("t")
    x_axis.set_ylabel("$X_t$")
    x_axis.get_yaxis().get_major_formatter().set_useOffset(False)

    v_axis.plot(v_mean, color="purple", label="mean value")
    v_axis.fill_between(
        list(range(len(v_mean))),
        v_mean - v_std,
        v_mean + v_std,
        alpha=0.3,
        color="purple",
        label="±$\sigma$",
    )
    v_axis.set_xlabel("t")
    v_axis.set_ylabel("$V_t$")
    v_axis.get_yaxis().get_major_formatter().set_useOffset(False)

    if q_mean.shape[0] < 10:
        ticks = [i for i in range(0, q_mean.shape[0])]

        q_axis.set_xticks(ticks)
        x_axis.set_xticks(ticks)
        v_axis.set_xticks(ticks)

    q_axis.legend()
    x_axis.legend()
    v_axis.legend()

    if save_mode:
        plt.savefig(f"{image_folder_name}/visualization_mean")
        plt.close()
    else:
        plt.show()

    # PLOT ALL STRATEGIES
    plt.figure()
    _, (q_axis, x_axis, v_axis) = plt.subplots(1, 3, figsize=(21, 7))

    for i, agent in enumerate(agents):
        Qs, Xs, Vs = sample_strategies(agent, env_function, args, info, n_test)

        q_mean = np.mean(Qs, axis=0)
        x_mean = np.mean(Xs, axis=0) / CENT_TO_DOLLAR
        v_mean = np.mean(Vs, axis=0) / CENT_TO_DOLLAR

        q_axis.plot(q_mean, label=f"run {i + 1}")
        x_axis.plot(x_mean, label=f"run {i + 1}")
        v_axis.plot(v_mean, label=f"run {i + 1}")

    q_axis.set_title("The inventory process - $Q_t$")
    x_axis.set_title("The cash process - $X_t$")
    v_axis.set_title("The value process - $V_t$")

    q_axis.set_xlabel("t")
    q_axis.set_ylabel("$Q_t$")
    q_axis.get_yaxis().get_major_formatter().set_useOffset(False)

    x_axis.set_xlabel("t")
    x_axis.set_ylabel("$X_t$")
    x_axis.get_yaxis().get_major_formatter().set_useOffset(False)

    v_axis.set_xlabel("t")
    v_axis.set_ylabel("$V_t$")
    v_axis.get_yaxis().get_major_formatter().set_useOffset(False)

    if q_mean.shape[0] < 10:
        ticks = [i for i in range(0, q_mean.shape[0])]

        q_axis.set_xticks(ticks)
        x_axis.set_xticks(ticks)
        v_axis.set_xticks(ticks)

    q_axis.legend()
    x_axis.legend()
    v_axis.legend()

    if save_mode:
        plt.savefig(f"{image_folder_name}/visualization_all")
        plt.close()
    else:
        plt.show()

    vec_env.close()


def plot_stability(
    agents, env, args, info, Q=2, save_mode=True, folder_name=None, n_resets=10
):
    """
    Plots three figures displaying the stability of DDQN runs.

    Parameters
    ----------
    agents : list
        a list of agents
    env : object
        a vec:_env
    args : dict
        a dict containing parameters
    info : dict
        a dict containing parameters
    Q : int
        the inventory levels the stabilty will be shown for (-Q,Q)
    save_mode : bool
        whether or not the figures should be saved or displayed
    folder_name : str
        where the figures will be saved
    n_resets : int
        how many resets that the values should be averaged over

    Returns
    -------
    None
    """
    mean_std_matrix = np.zeros((2 * Q + 1, int(args["T"] / args["dt"] + 1)))
    mean_unique_matrix = mean_std_matrix.copy()
    mean_error_matrix = mean_std_matrix.copy()

    suffix = f"_{int(n_resets * info['num_envs'])}"

    q_s = np.arange(-Q, Q + 1)
    t_s = np.arange(int(args["T"] / args["dt"] + 1)) * args["dt"]

    # For every combination of inventory level and time, find the best
    # action based on the given states

    statess = [env.reset() for _ in range(n_resets)]
    for n, states in enumerate(statess):
        std_matrix = np.zeros((2 * Q + 1, int(args["T"] / args["dt"] + 1)))
        unique_matrix = std_matrix.copy()
        error_matrix = std_matrix.copy()

        for i, q in enumerate(q_s):
            for j, t in enumerate(t_s):
                for state in states:
                    state[0] = q / 100
                    state[1] = t / args["T"]

                std_matrix[i, j] = get_standard_deviation(agents, states)
                unique_matrix[i, j] = get_n_unique_actions(agents, states)
                error_matrix[i, j] = get_n_errors(agents, states)

        mean_std_matrix += std_matrix
        mean_unique_matrix += unique_matrix
        mean_error_matrix += error_matrix

    mean_std_matrix = pd.DataFrame(mean_std_matrix / n_resets)
    mean_unique_matrix = pd.DataFrame(mean_unique_matrix / n_resets)
    mean_error_matrix = pd.DataFrame(mean_error_matrix / n_resets)

    mean_std_matrix = mean_std_matrix.set_axis(t_s, axis=1)
    mean_unique_matrix = mean_unique_matrix.set_axis(t_s, axis=1)
    mean_error_matrix = mean_error_matrix.set_axis(t_s, axis=1)

    mean_std_matrix.index = q_s
    mean_unique_matrix.index = q_s
    mean_error_matrix.index = q_s

    plt.figure()

    fig_std = sns.heatmap(mean_std_matrix)
    fig_std.set_title("Average standard deviation of optimal actions")
    fig_std.set_ylabel("inventory (q)")
    fig_std.set_xlabel("time (t)")

    if save_mode:
        plt.savefig(f"{folder_name}/heatmap_standard_deviation{suffix}")
        plt.close()
    else:
        plt.show()

    plt.figure()

    fig_unique = sns.heatmap(mean_unique_matrix, vmin=1, vmax=len(agents))
    fig_unique.set_title("Average number of unique optimal actions")
    fig_unique.set_ylabel("inventory (q)")
    fig_unique.set_xlabel("time (t)")

    if save_mode:
        plt.savefig(f"{folder_name}/heatmap_n_unique_actions{suffix}")
        plt.close()
    else:
        plt.show()

    plt.figure()

    fig_errors = sns.heatmap(mean_error_matrix, vmin=0, vmax=len(agents))
    fig_errors.set_title("Average number of actions not agreeing with mean strategy")
    fig_errors.set_ylabel("inventory (q)")
    fig_errors.set_xlabel("time (t)")

    if save_mode:
        plt.savefig(f"{folder_name}/heatmap_n_errors{suffix}")
        plt.close()
    else:
        plt.show()


def get_standard_deviation(agents, states):
    """
    Returns the standard deviation of optimal q-values for a given list of agents.

    Parameters
    ----------
    agents : list
        a list of agents
    states : list
        a list of states

    Returns
    -------
    q_std : np.array
        an array with average standard deviations
    """
    opt_actions = mean_opt_action([states], agents, batch=True)

    q_values = []

    for agent in agents:
        q_values.append(
            agent.get_q(states, False)[np.arange(len(opt_actions)), opt_actions]
        )

    q_std = np.mean(np.std(q_values, axis=0))

    return q_std


def get_n_errors(agents, states):
    """
    Returns the number of runs that doesnt agree with the mean strategy.

    Parameters
    ----------
    agents : list
        a list of agents
    states : list
        a list of states

    Returns
    -------
    n_errors_mean : np.array
        an array with the average number of errors
    """
    mean_opt_actions = mean_opt_action([states], agents, batch=True)

    n_errors = 0

    for agent in agents:
        opt_actions = np.argmax(agent.get_q(states, False), axis=1)

        n_errors += opt_actions != mean_opt_actions

    n_errors_mean = np.mean(n_errors)

    return n_errors_mean


def get_n_unique_actions(agents, states):
    """
    Returns the number of unqiue optimal actions.

    Parameters
    ----------
    agents : list
        a list of agents
    states : list
        a list of states

    Returns
    -------
    n_unique_mean : np.array
        an array with the average number of unique actions
    """
    opt_actions = []

    for agent in agents:
        opt_actions.append(np.argmax(agent.get_q(states, False), axis=1))

    opt_actions = np.array(opt_actions)

    n_unique = 0

    for col in range(opt_actions.shape[1]):
        n_unique += len(set(opt_actions[:, col]))

    n_unique_mean = n_unique / len(states)

    return n_unique_mean


# =================
# ===== BATCH =====
# =================


def setup_batch_env(args, env_function, num_envs=1, seed=0):
    """
    Setting up batch environment

    Parameters
    ----------
    info : dict containing info of environment
    env_function : function to create a single environment
    num_envs : number of environments to create
    seed : integer of seed

    Returns
    -------
    vec_env
    """
    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(num_envs) + seed * num_envs
    assert process_seeds.max() < 2**32

    # start = time.time()

    def make_env(idx):
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = process_seed
        env = env_function(args)
        env.seed(env_seed)
        return env

    vec_env = pfrl.envs.MultiprocessVectorEnv(
        [functools.partial(make_env, idx) for idx in range(num_envs)]
    )

    return vec_env


def get_env_function():
    """
    Returns a function used for creating an instance of the environment

    Parameters
    ----------
    None

    Returns
    -------
    env_function : fnc
        function used for creating an instance of the environment
    """

    def deep_env_function(args):
        """
        Returns an instance of the environment

        Parameters
        ----------
        args : dictionary
            parameters used for creating the instance

        Returns
        -------
        env : object
            an instance of the environment
        """
        from environments.mc_model.mc_environment_deep import MonteCarloEnvDeep

        EnvClass = MonteCarloEnvDeep
        env = EnvClass(**args)
        return env

    return deep_env_function
