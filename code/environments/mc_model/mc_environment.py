from abc import ABC
import pkg_resources
# pkg_resources.require("gym==0.22.0")
import gym
import numpy as np
import time
from environments.mc_model.lob_utils.lob_functions import LOB
from environments.mc_model.mc_lob_simulation_class import MarkovChainLobModel
import matplotlib.pyplot as plt
import torch as th
import pickle
import random


class MonteCarloEnv(gym.Env, ABC):
    """
    Parameters
    ----------
    include_spread_levels : bool
        argument passed on to LOB class, whether to include the spread levels in the LOB arrays
    ob_start : np.array
        the starting order book data. includes ask, spread, and initial volumes
    num_levels : int
        number of price levels to consider in LOB. num_levels = number of price levels above best bid and best ask to
        include
    max_quote_depth : int
        number of ticks away from best bid and best ask the MM can place its orders
    num_inv_buckets : int
        number of buckets to use on each side of the zeroth inventory bucket. total number of buckets is
        2*num_inv_buckets + 1
    T : int
        time horizon
    dt : float
        time step size
    mm_priority : bool
        whether MM's orders should have highest priority or not. note that False does not result in a time priority
        system, but instead the MM's orders have the last priority
    num_time_buckets : int
        nunmber of time variable state space bins
    kappa : int
        size of inventory buckets
    phi : float
        running inventory penalty parameter
    pre_run_on_start : bool
        whether to perform a pre-run as an object is created / the environment is reset
    pre_run_iterations : int
        number of simulation events to include in the pre-run
    simple_obs_space : bool
        TA BORT?
    MO_action : bool
        whether or not the MM can put MOs
    debug : bool
        whether or not information for debugging should be printed during simulation
    """

    def __init__(self, num_levels=10, include_spread_levels=True, max_quote_depth=5, num_inv_buckets=3,
                 T=5000, dt=1, mm_priority=True, num_time_buckets=5, kappa=3, phi=0, pre_run_on_start=False,
                 pre_run_iterations=int(1e4), simple_obs_space=True, MO_action=False, debug=False, randomize_reset = False,
                 default_order_size = 5, t_0 = True):

        super(MonteCarloEnv, self).__init__()

        self.include_spread_levels = include_spread_levels
        self.num_levels = num_levels
        self.starts_database = self.load_database()
        self.mc_model = MarkovChainLobModel(num_levels=self.num_levels, ob_start=self.starts_database[0],
                                            include_spread_levels=self.include_spread_levels)

        self.pre_run_on_start = pre_run_on_start
        self.pre_run_iterations = pre_run_iterations
        if self.pre_run_on_start:
            self.pre_run(self.pre_run_iterations)

        self.mm_priority = mm_priority
        self.num_time_buckets = num_time_buckets
        self.simple_obs_space = simple_obs_space
        self.MO_action = MO_action

        self.buy_mo_sizes = list()
        self.tot_vol_sell = list()
        self.sell_mo_sizes = list()
        self.tot_vol_buy = list()

        # Environment-defining variables
        self.max_quote_depth = max_quote_depth
        self.num_inv_buckets = num_inv_buckets
        self.kappa = kappa
        self.phi = phi
        self.T = T
        self.dt = dt
        self.t = 0
        self.t_0 = t_0

        # Agent and environment variables
        self.X_t = 0
        self.X_t_previous = 0
        self.Q_t = 0
        self.H_t = 0
        self.H_t_previous = 0
        self.default_order_size = default_order_size
        self.randomize_reset = randomize_reset

        self.num_errors = 0
        self.debug = debug

        # Initiating MM LO variables
        self.quotes_depths = {'bid': 1, 'ask': 1}
        self.quotes_absolute_level = {'bid': self.mc_model.ob.ask - self.quotes_depths['bid'],
                                      'ask': self.mc_model.ob.bid + self.quotes_depths['ask']}
        self.order_volumes = {'bid': 0, 'ask': 0}

        self.set_action_space()
        self.set_observation_space()

    def load_database(self, file_name = "environments/mc_model/starts_database/start_bank_n100000.pkl"):
        file = open(file_name, "rb")

        lob_data_base = pickle.load(file)

        return lob_data_base

    def set_action_space(self):
        """
        Sets the action space. The action space consists of the order depths and the action to place a market order.

        Minimum order depth is one. Maximum is max_quote_depth. Creates a gym.spaces.box.Box object for
        action_space['depths'].

        Binary coding indicating whether to post a market order 1/0. Creates a gym.spaces.discrete.Discrete object for
        action_space['market order'].

        Combines the quotes and market order into a dictionary (gym.spaces.dict.Dict object).
        """
        low = np.array([1, 1])
        high = np.array([self.max_quote_depth, self.max_quote_depth])
        spaces = {
            'depths': gym.spaces.Box(low=low, high=high, dtype=np.int16),
            'market order': gym.spaces.Discrete(2 if self.MO_action else 1)
        }
        self.action_space = gym.spaces.Dict(spaces)

    def _get_action_space_shape(self):
        """
        Returns the shape of the action space as a tuple
        """

        depths = self.action_space["depths"].high - self.action_space["depths"].low + 1
        MO = self.action_space["market order"].n
        return tuple(np.append(depths, MO))

    def set_observation_space(self):
        """
        Creates the observation space
        """
        if self.simple_obs_space:
            self.observation_space = gym.spaces.Box(low=np.array([-self.num_inv_buckets, 1]),
                                                    high=np.array([self.num_inv_buckets, self.num_time_buckets]),
                                                    dtype=np.int16)
        else:
            obs_dim = 2 + 2 * self.num_levels
            low = -np.inf * np.ones(obs_dim)
            high = np.inf * np.ones(obs_dim)
            self.observation_space = gym.spaces.Box(low=low, high=high)
        """spaces = {
            'spread': gym.spaces.Box(low=1, high=5, shape=(1,), dtype=np.int16),
            't': gym.spaces.Box(low=0, high=self.T/self.num_time_buckets, shape=(1,), dtype=np.int16),
            'inventory': gym.spaces.Box(low=-self.max_Q, high=self.max_Q, shape=(1,), dtype=np.int16),
            'q_bid': gym.spaces.Box(low=0, high=50, shape=(self.num_levels,), dtype=np.int16),
            'q_ask': gym.spaces.Box(low=0, high=50, shape=(self.num_levels,), dtype=np.int16)
        }
        self.observation_space = gym.spaces.Dict(spaces)"""

    def state(self):
        """
        Returns the current observable state


        OLD
        # Creates an array of (0.2, 0.4, 0.6, 0.8) (for num_time_steps=5) which are the lower thresholds for each
            # time interval
            time_thresholds = np.linspace(0, 1 - 1/self.num_time_steps, self.num_time_steps)
            # Finds the current time interval
            current_time_interval = int(len(time_thresholds) - np.argmax(np.flip(self.t/self.T >= time_thresholds)))


        Returns
        -------
        obs : tuple
            the observation space in terms of (time_varible, inventory_variable)
        """

        # inventory_variable = np.min(
        #     [np.ceil(self.Q_t / self.kappa * self.num_inv_buckets), self.num_inv_buckets]) * (
        #                                  self.Q_t > 0) + np.max(
        #     [np.floor(self.Q_t / self.kappa * self.num_inv_buckets), -self.num_inv_buckets]) * (self.Q_t < 0)

        # New inventory variable
        inventory_variable = np.min([np.ceil(self.Q_t / self.kappa), self.num_inv_buckets]) * (self.Q_t > 0) + \
                             np.max([np.floor(self.Q_t / self.kappa), -self.num_inv_buckets]) * (self.Q_t < 0)

        time_variable = np.ceil(self.t / self.T * self.num_time_buckets)

        if self.t_0:
            time_variable += self.t == 0

        return int(inventory_variable), int(time_variable)

    def state_deep(self):
        lob = self.mc_model.ob.data[:, 1:].flatten()
        inv = self.Q_t
        t = self.t
        state = tuple(np.concatenate([np.array([inv, t]), lob]))
        return th.from_numpy(np.asarray(state)).float().unsqueeze(0)

    def step(self, action: dict):
        """

        :param action:
        :return:
        """

        self.X_t_previous = self.X_t
        self.H_t_previous = self.H_t

        if self.t < self.T - self.dt:

            # --------------- Market order execution ---------------

            if action['market order'] == 1:
                # The market maker can place a MO with the volume equal to the volume available on the best bid/ask.
                # Other market participants cannot walk the book so it's only fair that the MM cannot do it either in
                # 'one shot'.

                if self.Q_t > 0:  # Decreasing net inventory
                    volume_to_sell_current_trade = np.min([self.Q_t, self.mc_model.ob.bid_volume])
                    self.X_t += self.mc_model.ob.sell_n(volume_to_sell_current_trade)
                    self.mc_model.ob.change_volume(level=self.mc_model.ob.bid, absolute_level=True,
                                                   volume=-volume_to_sell_current_trade)
                    self.Q_t -= volume_to_sell_current_trade

                elif self.Q_t < 0:  # Increasing net inventory
                    volume_to_buy_current_trade = np.min([-self.Q_t, self.mc_model.ob.ask_volume])
                    self.X_t -= self.mc_model.ob.buy_n(volume_to_buy_current_trade)
                    self.mc_model.ob.change_volume(level=self.mc_model.ob.ask, absolute_level=True,
                                                   volume=volume_to_buy_current_trade)
                    self.Q_t += volume_to_buy_current_trade

            # --------------- Placing limit orders in the order book ---------------
            mm_bid_depth, mm_ask_depth = action['depths']
            self.order_volumes['bid'] = self.default_order_size
            self.order_volumes['ask'] = self.default_order_size

            # We need to specify the bid and ask prices before we start placing orders, otherwise we might alter them
            ask = self.mc_model.ob.ask
            bid = self.mc_model.ob.bid

            # If invalid depths make adjustments
            if self.mc_model.ob.spread >= mm_bid_depth + mm_ask_depth:
                diff = self.mc_model.ob.spread - (mm_ask_depth + mm_bid_depth) + 1
                mm_bid_depth += int(np.ceil(diff / 2))
                mm_ask_depth += int(np.ceil(diff / 2))

            # Placing the orders
            self.quotes_absolute_level['bid'] = ask - mm_bid_depth
            self.quotes_absolute_level['ask'] = bid + mm_ask_depth
            self.mc_model.ob.change_volume(level=ask - mm_bid_depth, volume=-self.default_order_size,
                                           absolute_level=True)
            self.mc_model.ob.change_volume(level=bid + mm_ask_depth, volume=self.default_order_size,
                                           absolute_level=True)

            # The total volumes on market maker's bid and ask levels
            # IMPORTANT: these quantities must be computed BEFORE simulation occurs
            vol_on_mm_bid = -self.mc_model.ob.get_volume(self.quotes_absolute_level['bid'], absolute_level=True)
            vol_on_mm_ask = self.mc_model.ob.get_volume(self.quotes_absolute_level['ask'], absolute_level=True)

        else:
            # --------------- Liquidating the inventory ---------------

            if self.debug:
                print("Liquidating the inventory")
                print(self.mc_model.ob.data)
                print(f"The bid volumes: {self.mc_model.ob.q_bid()}")
                print(f"The ask volumes: {self.mc_model.ob.q_ask()}")
                print(f"MM's inventory before liquidation: {self.Q_t}")

            can_trade = True
            if self.Q_t > 0:
                level = self.mc_model.ob.bid
            else:
                level = self.mc_model.ob.ask
            while can_trade:
                # N.B. If the while-loop is traversed through more than once it means that the market maker has to
                # walk the book in order to liquidate all of its holdings
                if self.Q_t > 0:
                    volume_to_sell_current_trade = np.min([self.Q_t, self.mc_model.ob.bid_volume])
                    if self.mc_model.ob.bid_volume == 0 and self.mc_model.ob.outside_volume != 0:  # self.Q_t != 0 and
                        volume_to_sell_current_trade = self.mc_model.ob.outside_volume
                    elif self.mc_model.ob.bid_volume == 0 and self.mc_model.ob.outside_volume == 0:
                        can_trade = False
                    if can_trade:
                        if np.sum(self.mc_model.ob.q_bid()) == 0:
                            trade_turnover = level * self.mc_model.ob.outside_volume
                        else:
                            trade_turnover = self.mc_model.ob.sell_n(volume_to_sell_current_trade)
                            self.mc_model.ob.change_volume(level=self.mc_model.ob.bid, absolute_level=True,
                                                           volume=volume_to_sell_current_trade)
                        level -= 1
                        self.X_t += trade_turnover
                        self.Q_t -= volume_to_sell_current_trade
                        if self.debug:
                            print(f"Selling {volume_to_sell_current_trade} for {trade_turnover}, "
                                  f"remaining inventory {self.Q_t}")
                elif self.Q_t < 0:
                    volume_to_buy_current_trade = np.min([-self.Q_t, self.mc_model.ob.ask_volume])
                    if self.mc_model.ob.ask_volume == 0 and self.mc_model.ob.outside_volume != 0:  # and self.Q_t != 0
                        volume_to_buy_current_trade = self.mc_model.ob.outside_volume
                    elif self.mc_model.ob.ask_volume == 0 and self.mc_model.ob.outside_volume == 0:
                        can_trade = False
                    if can_trade:
                        if np.sum(self.mc_model.ob.q_ask()) == 0:
                            trade_turnover = level * self.mc_model.ob.outside_volume
                        else:
                            trade_turnover = self.mc_model.ob.buy_n(volume_to_buy_current_trade)
                            self.mc_model.ob.change_volume(level=self.mc_model.ob.ask, absolute_level=True,
                                                           volume=-volume_to_buy_current_trade)
                        level += 1
                        self.X_t -= trade_turnover
                        self.Q_t += volume_to_buy_current_trade
                        if self.debug:
                            print(f"Buying {volume_to_buy_current_trade} for {trade_turnover}, "
                                  f"remaining inventory {self.Q_t}")
                else:
                    can_trade = False

            if self.debug:
                print("Liquidation done!")
                print(f"The bid volumes: {self.mc_model.ob.q_bid()}")
                print(f"The ask volumes: {self.mc_model.ob.q_ask()}")
                print(f"MM's inventory after liquidation: {self.Q_t}")

        # --------------- Simulating the environment ---------------

        # TODO: simulate lob events and investigate if MM's outstanding orders have been filled. If so adjust inventory

        self.t += self.dt  # increase the current time stamp

        # Simulating the LOB
        if self.t >= self.T:
            simulation_results = self.mc_model.simulate(end_time=self.dt)
        else:
            simulation_results = self.mc_model.simulate(end_time=self.dt, order_volumes=self.order_volumes,
                                                        order_prices=self.quotes_absolute_level)

        # We are only interested in the actual simulation results before the terminal time step since we have no
        # outstanding limit orders by then
        if self.t < self.T:

            # Market maker's absolute price levels
            mm_bid_abs = self.quotes_absolute_level["bid"]
            mm_ask_abs = self.quotes_absolute_level["ask"]

            # To manage cancellations - NOT NEEDED ANYMORE
            has_bought = 0
            has_sold = 0

            # Looping through all simulated results
            for n in range(simulation_results['num_events']):
                event = simulation_results['event'][n]
                absolute_level = simulation_results['abs_level'][n]
                size = simulation_results['size'][n]

                # print(self.order_volumes)

                # Arriving LO buy or LO buy cancellation
                if event in [0, 4] and absolute_level == mm_bid_abs:
                    if event == 0:  # LO buy
                        vol_on_mm_bid += size
                        # self.order_volumes["bid"] = np.max([np.min([vol_on_mm_bid, 5 - has_bought]), 0])
                    else:  # LO buy cancellation
                        vol_on_mm_bid -= size
                        """if vol_on_mm_bid < self.order_volumes["bid"]:
                            self.order_volumes["bid"] = vol_on_mm_bid"""

                # Arriving LO sell or LO sell cancellation
                elif event in [1, 5] and absolute_level == mm_ask_abs:
                    if event == 1:  # LO sell
                        vol_on_mm_ask += size
                        # self.order_volumes["ask"] = np.max([np.min([vol_on_mm_ask, 5 - has_sold]), 0])
                    else:  # LO sell cancellation
                        vol_on_mm_ask -= size
                        """if vol_on_mm_ask < self.order_volumes["ask"]:
                            self.order_volumes["ask"] = vol_on_mm_ask"""

                # MM's orders only affected by MOs
                elif event in [2, 3] and self.t != self.T:
                    # event type 2: 'mo bid', i.e., MO sell order arrives
                    if event == 2 and absolute_level == self.quotes_absolute_level['bid']:
                        if not self.mm_priority:
                            # IMPLICIT ASSUMPTION THAT MARKET MAKER HAS LAST ORDER PRIORITY
                            if vol_on_mm_bid - size < self.order_volumes['bid']:
                                mm_trade_volume = size - (vol_on_mm_bid - self.order_volumes['bid'])
                                self.order_volumes['bid'] -= mm_trade_volume  # decrease outstanding LO volume
                                self.X_t -= mm_trade_volume * absolute_level  # deduct cash
                                self.Q_t += mm_trade_volume  # adjust inventory
                                # has_bought += mm_trade_volume
                        else:
                            # IMPLICIT ASSUMPTION THAT MARKET MAKER HAS FIRST ORDER PRIORITY
                            mm_trade_volume = np.min([size, self.order_volumes['bid']])
                            self.order_volumes['bid'] -= mm_trade_volume
                            self.X_t -= mm_trade_volume * absolute_level
                            self.Q_t += mm_trade_volume
                            # has_bought += mm_trade_volume
                        vol_on_mm_bid -= size

                    # event type 3: 'mo ask', i.e., MO buy order arrives
                    elif event == 3 and absolute_level == self.quotes_absolute_level['ask']:
                        if not self.mm_priority:
                            # IMPLICIT ASSUMPTION THAT MARKET MAKER HAS LAST ORDER PRIORITY
                            if vol_on_mm_ask - size < self.order_volumes['ask']:
                                mm_trade_volume = size - (vol_on_mm_ask - self.order_volumes['ask'])
                                self.order_volumes['ask'] -= mm_trade_volume  # decrease outstanding LO volume
                                self.X_t += mm_trade_volume * absolute_level  # increase cash
                                self.Q_t -= mm_trade_volume  # adjust inventory
                                # has_sold += mm_trade_volume
                        else:
                            # IMPLICIT ASSUMPTION THAT MARKET MAKER HAS FIRST ORDER PRIORITY
                            mm_trade_volume = np.min([size, self.order_volumes['ask']])
                            self.order_volumes['ask'] -= mm_trade_volume
                            self.X_t += mm_trade_volume * absolute_level
                            self.Q_t -= mm_trade_volume
                            # has_sold += mm_trade_volume
                        vol_on_mm_ask -= size

        # self.print_simulation_results(simulation_results)
        # TODO : adjust X_t_previous if MM places MO
        if self.debug:
            if not self.MO_action:
                if self.X_t != self.print_MO_results(simulation_results, self.X_t_previous, printing=False) \
                        and self.t < self.T:
                    self.num_errors += 1
                    self.print_MO_results(simulation_results, self.X_t_previous, printing=True)
                    print("--> should have been", self.X_t, "at time", self.t)
                    input("Next:")

        # TODO: cancel MM's outstanding
        # If the market maker has an outstanding buy order, cancel the entire volume
        if self.t != self.T:
            if self.order_volumes['bid'] > 0:
                self.mc_model.ob.change_volume(level=self.quotes_absolute_level['bid'],
                                               volume=np.min([self.order_volumes['bid'],
                                                              -self.mc_model.ob.get_volume(
                                                                  self.quotes_absolute_level['bid'],
                                                                  absolute_level=True)]),
                                               absolute_level=True)

            # If the market maker has an outstanding sell order, cancel the entire volume
            if self.order_volumes['ask'] > 0:
                self.mc_model.ob.change_volume(level=self.quotes_absolute_level['ask'],
                                               volume=-np.min([self.order_volumes['ask'],
                                                               self.mc_model.ob.get_volume(
                                                                   self.quotes_absolute_level['ask'],
                                                                   absolute_level=True)]),
                                               absolute_level=True)

        if self.Q_t > 0:
            self.H_t = self.mc_model.ob.sell_n(self.Q_t)
        else:
            self.H_t = -self.mc_model.ob.buy_n(-self.Q_t)

        if self.simple_obs_space:
            return self.state(), self._get_reward()
        else:
            return self.state_deep(), self._get_reward()

    def _get_reward(self):
        # TODO: different rewards for final step and intermediate steps

        # Added running inventory penalty
        return self.X_t + self.H_t - (self.X_t_previous + self.H_t_previous) - self.phi * self.Q_t ** 2

    def pre_run(self, n_steps=int(1e4)):
        """
        Simulates n_steps events / transitions in the Markov chain LOB

        Parameters
        ----------
        n_steps : int
            the number of events that should be simulated in the LOB
        """
        self.mc_model.simulate(int(n_steps))

    def reset(self, randomized = None):
        self.X_t = 0
        self.t = 0
        self.Q_t = 0

        # Ugly but for efficiency
        if randomized == None and self.randomize_reset:
            ob_start = random.choice(self.starts_database)
        elif randomized:
            ob_start = random.choice(self.starts_database)
        else:
            ob_start = self.starts_database[0]
        
        self.mc_model = MarkovChainLobModel(include_spread_levels=self.include_spread_levels,
                                            ob_start=ob_start, num_levels=self.num_levels)
        if self.pre_run_on_start:
            self.pre_run()


    def render(self, mode='human'):
        print('=' * 40)
        print(f'End of t = {self.t}')
        print(f'Current bid-ask = {self.mc_model.ob.bid}-{self.mc_model.ob.ask}')
        print(f'Current inventory = {self.Q_t}')
        print(f'Cash value = {self.X_t}')
        print(f'Holding value = {self.H_t}')
        print(f'Total value = {self.X_t + self.H_t}')
        print(f'State = {self.state()}')
        print('=' * 40 + '\n')

    def print_simulation_results(self, simulation_results):
        for key in simulation_results.keys():
            if key == "event":
                print("\n event : ", end="")
                for event in simulation_results[key]:
                    print(self.mc_model.inverse_event_types[event], end=", ")
            else:
                print("\n", key, ":", simulation_results[key], end="")

    def print_MO_results(self, simulation_results, cash, printing=True):
        if 2 in simulation_results["event"] or 3 in simulation_results["event"]:
            if printing:
                print("=" * 40)
                self.print_simulation_results(simulation_results)
                print("\n" + "-" * 40)
                print("t =", self.t)
                print(self.order_volumes)
                print(self.quotes_absolute_level)
            size_bid = self.default_order_size
            size_ask = self.default_order_size
            for i, event in enumerate(simulation_results["event"]):
                size_bid_event = min(simulation_results["size"][i], size_bid)
                size_ask_event = min(simulation_results["size"][i], size_ask)
                level = simulation_results["abs_level"][i]
                if self.mc_model.inverse_event_types[event] == "mo bid" and level == self.quotes_absolute_level["bid"]:
                    if printing:
                        print("MO sell, size", str(size_bid_event) + ", price", level)
                        print("cash:", cash, "-->", cash, "-", size_bid_event, "*", level, "=",
                              cash - size_bid_event * level)
                    cash -= size_bid_event * level
                    size_bid -= size_bid_event
                elif self.mc_model.inverse_event_types[event] == "mo ask" \
                        and level == self.quotes_absolute_level["ask"]:
                    if printing:
                        print("MO buy, size", str(size_ask_event) + ", price", level)
                        print("cash:", cash, "-->", cash, "+", size_ask_event, "*", level, "=",
                              cash + size_ask_event * level)
                    cash += size_ask_event * level
                    size_ask -= size_ask_event
            if printing:
                print("=" * 40)
        return cash


def init_LOB(num_levels=10, init_ask=10000, init_spread=1, init_volume=1):
    lob = np.zeros((2, num_levels + 1), dtype=int)
    lob[0, 0] = init_ask
    lob[1, 0] = -init_spread
    lob[0, init_spread:] = init_volume
    lob[1, init_spread:] = -init_volume
    return lob


def showcase_plot(mid_prices, Q_t, X_t, include_legends=True):
    fig, (price, inventory, cash) = plt.subplots(1, 3, figsize=(21, 7))

    price.set_title("mid price")
    inventory.set_title("Q_t")
    cash.set_title("X_t")

    for i in range(len(mid_prices)):
        price.plot(mid_prices[i], label="run " + str(i + 1))
        inventory.plot(Q_t[i], label="run " + str(i + 1))
        cash.plot(X_t[i], label="run " + str(i + 1))

    if include_legends:
        price.legend()
        inventory.legend()
        cash.legend()

    plt.show()


def showcase_plot_CI(mid_prices, Q_t, X_t, include_legends=True):
    fig, (price, inventory, cash) = plt.subplots(1, 3, figsize=(21, 7))

    price.set_title("mid price")
    inventory.set_title("Q_t")
    cash.set_title("X_t")

    for i in range(len(mid_prices)):
        price.plot(mid_prices[i], label="run " + str(i + 1))
        inventory.plot(Q_t[i], label="run " + str(i + 1))
        cash.plot(X_t[i], label="run " + str(i + 1))

    if include_legends:
        price.legend()
        inventory.legend()
        cash.legend()

    plt.show()


def mid_price_comparison(mid_prices_natural, mid_prices_mm_sym, mid_prices_mm_asym):
    """
    Function to plot a comparison between different market making strategies
    """

    fig, (natural, mm_sym, mm_asymmetrical) = plt.subplots(1, 3, figsize=(21, 7))

    natural.set_title("No market making")
    mm_sym.set_title("Symmetrical market making")
    mm_asymmetrical.set_title("Asymmetrical market making")

    # natural_mid_min = np.min(mid_prices_natural, axis=0)
    natural_mid_std = np.std(mid_prices_natural, axis=0)
    natural_mid_mean = np.mean(mid_prices_natural, axis=0)
    # natural_mid_max = np.max(mid_prices_natural, axis=0)

    # sym_mid_min = np.min(mid_prices_mm_sym, axis=0)
    sym_mid_mean = np.mean(mid_prices_mm_sym, axis=0)
    sym_mid_std = np.std(mid_prices_mm_sym, axis=0)
    # sym_mid_max = np.max(mid_prices_mm_sym, axis=0)

    # asym_mid_min = np.min(mid_prices_mm_asym, axis=0)
    asym_mid_mean = np.mean(mid_prices_mm_asym, axis=0)
    asym_mid_std = np.std(mid_prices_mm_asym, axis=0)
    # asym_mid_max = np.max(mid_prices_mm_asym, axis=0)

    # natural.fill_between(list(range(len(natural_mid_min))), natural_mid_min, natural_mid_max, alpha = 0.3, color="purple")
    natural.plot(natural_mid_mean, color="purple")
    natural.fill_between(list(range(len(natural_mid_mean))), natural_mid_mean - natural_mid_std, natural_mid_mean + natural_mid_std, alpha=0.3, color="purple")
    print(f'std natural = {natural_mid_std[-1]}')
    natural.set_xlabel("t")
    natural.set_ylabel("price")
    natural.set_ylabel("mid price")
    natural.get_yaxis().get_major_formatter().set_useOffset(False)
    mm_sym.set_ylim([np.min([np.min(sym_mid_mean[:-1] - sym_mid_std[:-1]), np.min(natural_mid_mean - natural_mid_std)]), np.max([np.max(sym_mid_mean[:-1] + sym_mid_std[:-1]), np.max(natural_mid_mean + natural_mid_std)])])

    mm_sym.plot(sym_mid_mean[:-1], color="purple")
    # mm_sym.fill_between(list(range(len(sym_mid_mean))), sym_mid_min, sym_mid_max, alpha=0.3, color="purple")
    mm_sym.fill_between(list(range(len(sym_mid_mean[:-1]))), sym_mid_mean[:-1] - sym_mid_std[:-1], sym_mid_mean[:-1] + sym_mid_std[:-1], alpha=0.3, color="purple")
    mm_sym.set_xlabel("t")
    print(f'std mm sym = {sym_mid_std[-2]}')
    mm_sym.set_ylabel("price")
    mm_sym.set_ylabel("mid price")
    mm_sym.get_yaxis().get_major_formatter().set_useOffset(False)
    mm_sym.set_ylim([np.min([np.min(sym_mid_mean[:-1] - sym_mid_std[:-1]), np.min(natural_mid_mean - natural_mid_std)]), np.max([np.max(sym_mid_mean[:-1] + sym_mid_std[:-1]), np.max(natural_mid_mean + natural_mid_std)])])

    mm_asymmetrical.plot(asym_mid_mean[:-1], color="purple")
    # mm_asymmetrical.fill_between(list(range(len(asym_mid_mean))), asym_mid_min, asym_mid_max, alpha=0.3, color="purple")
    mm_asymmetrical.fill_between(list(range(len(asym_mid_mean[:-1]))), asym_mid_mean[:-1] - asym_mid_std[:-1], asym_mid_mean[:-1] + asym_mid_std[:-1], alpha=0.3, color="purple")
    mm_asymmetrical.set_xlabel("t")
    mm_asymmetrical.set_ylabel("price")
    mm_asymmetrical.set_ylabel("mid price")
    mm_asymmetrical.get_yaxis().get_major_formatter().set_useOffset(False)


if __name__ == '__main__':
    x0 = init_LOB(num_levels=10, init_ask=10001, init_spread=2, init_volume=1)
    env = MonteCarloEnv(include_spread_levels=True, num_levels=10, mm_priority=True, T=int(1e4),
                        num_time_buckets=10)

    mid_prices = []
    Q_ts = []
    X_ts = []

    for episode in range(1):
        mid_prices.append([])
        Q_ts.append([])
        X_ts.append([])

        env.reset()
        t = time.time()
        tot_reward = 0
        while env.t < env.T:
            reward = env.step({'depths': np.array([1, 1], dtype='int16'), 'market order': 0})
            # print(f'Reward: {reward}')
            # tot_reward += reward
            # print(f'Tot reward: {tot_reward}')
            env.render()

            mid_prices[episode].append(env.mc_model.ob.mid / 100)
            Q_ts[episode].append(env.Q_t)
            X_ts[episode].append(env.X_t / 100)
        print(time.time() - t)
        # print(env.num_errors)

    # showcase_plot(mid_prices, Q_ts, X_ts)
