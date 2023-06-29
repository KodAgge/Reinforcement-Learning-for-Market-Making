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
    num_levels : int
        number of price levels to consider in LOB. num_levels = number of price levels above best bid and best ask to
        include
    include_spread_levels : bool
        argument passed on to LOB class, whether to include the spread levels in the LOB arrays
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
    MO_action : bool
        whether or not the MM can put MOs
    debug : bool
        whether or not information for debugging should be printed during simulation
    randomize_reset : bool
        if the LOB should get a random state after every reset
    default_order_size : int
        the size of the orders the MM places
    t_0 : bool
        if the time should start on 0 or 1

    Returns
    -------
    None
    """

    def __init__(
        self,
        num_levels=10,
        include_spread_levels=True,
        max_quote_depth=5,
        num_inv_buckets=3,
        T=5000,
        dt=1,
        mm_priority=True,
        num_time_buckets=5,
        kappa=3,
        phi=0,
        pre_run_on_start=False,
        pre_run_iterations=int(1e4),
        MO_action=False,
        debug=False,
        randomize_reset=False,
        default_order_size=5,
        t_0=True,
    ):
        super(MonteCarloEnv, self).__init__()

        self.include_spread_levels = include_spread_levels
        self.num_levels = num_levels
        self.starts_database = self.load_database()
        self.mc_model = MarkovChainLobModel(
            num_levels=self.num_levels,
            ob_start=self.starts_database[0],
            include_spread_levels=self.include_spread_levels,
        )

        self.pre_run_on_start = pre_run_on_start
        self.pre_run_iterations = pre_run_iterations
        if self.pre_run_on_start:
            self.pre_run(self.pre_run_iterations)

        self.mm_priority = mm_priority
        self.num_time_buckets = num_time_buckets
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
        self.quotes_depths = {"bid": 1, "ask": 1}
        self.quotes_absolute_level = {
            "bid": self.mc_model.ob.ask - self.quotes_depths["bid"],
            "ask": self.mc_model.ob.bid + self.quotes_depths["ask"],
        }
        self.order_volumes = {"bid": 0, "ask": 0}

        self.set_action_space()
        self.set_observation_space()

    def load_database(
        self, file_name="environments/mc_model/starts_database/start_bank_n100000.pkl"
    ):
        """
        loads a database with random LOB states

        Parameters
        ----------
        file_name : str
            where the database is stored

        Returns
        -------
        lob_data_base : list
            a list with random LOB states
        """

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

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        low = np.array([1, 1])
        high = np.array([self.max_quote_depth, self.max_quote_depth])
        spaces = {
            "depths": gym.spaces.Box(low=low, high=high, dtype=np.int16),
            "market order": gym.spaces.Discrete(2 if self.MO_action else 1),
        }
        self.action_space = gym.spaces.Dict(spaces)

    def _get_action_space_shape(self):
        """
        Returns the shape of the action space as a tuple

        Parameters
        ----------
        None

        Returns
        -------
        shape : tuple
            the shape of the action space
        """

        depths = self.action_space["depths"].high - self.action_space["depths"].low + 1
        MO = self.action_space["market order"].n
        return tuple(np.append(depths, MO))

    def set_observation_space(self):
        """
        Creates the observation space

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.observation_space = gym.spaces.Box(
            low=np.array([-self.num_inv_buckets, 1]),
            high=np.array([self.num_inv_buckets, self.num_time_buckets]),
            dtype=np.int16,
        )

    def state(self):
        """
        Returns the current observable state

        Parameters
        ----------
        None

        Returns
        -------
        obs : tuple
            the observation space in terms of (time_varible, inventory_variable)
        """

        # New inventory variable
        inventory_variable = np.min(
            [np.ceil(self.Q_t / self.kappa), self.num_inv_buckets]
        ) * (self.Q_t > 0) + np.max(
            [np.floor(self.Q_t / self.kappa), -self.num_inv_buckets]
        ) * (
            self.Q_t < 0
        )

        time_variable = np.ceil(self.t / self.T * self.num_time_buckets)

        if self.t_0:
            time_variable += self.t == 0

        return int(inventory_variable), int(time_variable)

    def step(self, action: dict):
        """
        Receives an action (bid and ask depths) and takes a step in the environment accordingly

        Parameters
        ----------
        action : dict
            the bid and ask depths

        Returns
        -------
        obs : tuple
            the observation space in terms of (time_varible, inventory_variable)
        reward : float
            the reward received at time t+1
        """

        self.X_t_previous = self.X_t
        self.H_t_previous = self.H_t

        if self.t < self.T - self.dt:
            # --------------- Placing limit orders in the order book ---------------
            mm_bid_depth, mm_ask_depth = action["depths"]
            self.order_volumes["bid"] = self.default_order_size
            self.order_volumes["ask"] = self.default_order_size

            # We need to specify the bid and ask prices before we start placing orders, otherwise we might alter them
            ask = self.mc_model.ob.ask
            bid = self.mc_model.ob.bid

            # If invalid depths make adjustments
            if self.mc_model.ob.spread >= mm_bid_depth + mm_ask_depth:
                diff = self.mc_model.ob.spread - (mm_ask_depth + mm_bid_depth) + 1
                mm_bid_depth += int(np.ceil(diff / 2))
                mm_ask_depth += int(np.ceil(diff / 2))

            # Placing the orders
            self.quotes_absolute_level["bid"] = ask - mm_bid_depth
            self.quotes_absolute_level["ask"] = bid + mm_ask_depth
            self.mc_model.ob.change_volume(
                level=ask - mm_bid_depth,
                volume=-self.default_order_size,
                absolute_level=True,
            )
            self.mc_model.ob.change_volume(
                level=bid + mm_ask_depth,
                volume=self.default_order_size,
                absolute_level=True,
            )

            # The total volumes on market maker's bid and ask levels
            # IMPORTANT: these quantities must be computed BEFORE simulation occurs
            vol_on_mm_bid = -self.mc_model.ob.get_volume(
                self.quotes_absolute_level["bid"], absolute_level=True
            )
            vol_on_mm_ask = self.mc_model.ob.get_volume(
                self.quotes_absolute_level["ask"], absolute_level=True
            )

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
                    volume_to_sell_current_trade = np.min(
                        [self.Q_t, self.mc_model.ob.bid_volume]
                    )
                    if (
                        self.mc_model.ob.bid_volume == 0
                        and self.mc_model.ob.outside_volume != 0
                    ):
                        volume_to_sell_current_trade = self.mc_model.ob.outside_volume
                    elif (
                        self.mc_model.ob.bid_volume == 0
                        and self.mc_model.ob.outside_volume == 0
                    ):
                        can_trade = False
                    if can_trade:
                        if np.sum(self.mc_model.ob.q_bid()) == 0:
                            trade_turnover = level * self.mc_model.ob.outside_volume
                        else:
                            trade_turnover = self.mc_model.ob.sell_n(
                                volume_to_sell_current_trade
                            )
                            self.mc_model.ob.change_volume(
                                level=self.mc_model.ob.bid,
                                absolute_level=True,
                                volume=volume_to_sell_current_trade,
                            )
                        level -= 1
                        self.X_t += trade_turnover
                        self.Q_t -= volume_to_sell_current_trade
                        if self.debug:
                            print(
                                f"Selling {volume_to_sell_current_trade} for {trade_turnover}, "
                                f"remaining inventory {self.Q_t}"
                            )
                elif self.Q_t < 0:
                    volume_to_buy_current_trade = np.min(
                        [-self.Q_t, self.mc_model.ob.ask_volume]
                    )
                    if (
                        self.mc_model.ob.ask_volume == 0
                        and self.mc_model.ob.outside_volume != 0
                    ):
                        volume_to_buy_current_trade = self.mc_model.ob.outside_volume
                    elif (
                        self.mc_model.ob.ask_volume == 0
                        and self.mc_model.ob.outside_volume == 0
                    ):
                        can_trade = False
                    if can_trade:
                        if np.sum(self.mc_model.ob.q_ask()) == 0:
                            trade_turnover = level * self.mc_model.ob.outside_volume
                        else:
                            trade_turnover = self.mc_model.ob.buy_n(
                                volume_to_buy_current_trade
                            )
                            self.mc_model.ob.change_volume(
                                level=self.mc_model.ob.ask,
                                absolute_level=True,
                                volume=-volume_to_buy_current_trade,
                            )
                        level += 1
                        self.X_t -= trade_turnover
                        self.Q_t += volume_to_buy_current_trade
                        if self.debug:
                            print(
                                f"Buying {volume_to_buy_current_trade} for {trade_turnover}, "
                                f"remaining inventory {self.Q_t}"
                            )
                else:
                    can_trade = False

            if self.debug:
                print("Liquidation done!")
                print(f"The bid volumes: {self.mc_model.ob.q_bid()}")
                print(f"The ask volumes: {self.mc_model.ob.q_ask()}")
                print(f"MM's inventory after liquidation: {self.Q_t}")

        # --------------- Simulating the environment ---------------
        self.t += self.dt  # increase the current time stamp

        # Simulating the LOB
        if self.t >= self.T:
            simulation_results = self.mc_model.simulate(end_time=self.dt)
        else:
            simulation_results = self.mc_model.simulate(
                end_time=self.dt,
                order_volumes=self.order_volumes,
                order_prices=self.quotes_absolute_level,
            )

        # We are only interested in the actual simulation results before the terminal time step since we have no
        # outstanding limit orders by then
        if self.t < self.T:
            # Market maker's absolute price levels
            mm_bid_abs = self.quotes_absolute_level["bid"]
            mm_ask_abs = self.quotes_absolute_level["ask"]

            # Looping through all simulated results
            for n in range(simulation_results["num_events"]):
                event = simulation_results["event"][n]
                absolute_level = simulation_results["abs_level"][n]
                size = simulation_results["size"][n]

                # Arriving LO buy or LO buy cancellation
                if event in [0, 4] and absolute_level == mm_bid_abs:
                    if event == 0:  # LO buy
                        vol_on_mm_bid += size

                    else:  # LO buy cancellation
                        vol_on_mm_bid -= size

                # Arriving LO sell or LO sell cancellation
                elif event in [1, 5] and absolute_level == mm_ask_abs:
                    if event == 1:  # LO sell
                        vol_on_mm_ask += size

                    else:  # LO sell cancellation
                        vol_on_mm_ask -= size

                # MM's orders only affected by MOs
                elif event in [2, 3] and self.t != self.T:
                    # event type 2: 'mo bid', i.e., MO sell order arrives
                    if (
                        event == 2
                        and absolute_level == self.quotes_absolute_level["bid"]
                    ):
                        if not self.mm_priority:
                            # IMPLICIT ASSUMPTION THAT MARKET MAKER HAS LAST ORDER PRIORITY
                            if vol_on_mm_bid - size < self.order_volumes["bid"]:
                                mm_trade_volume = size - (
                                    vol_on_mm_bid - self.order_volumes["bid"]
                                )
                                self.order_volumes[
                                    "bid"
                                ] -= mm_trade_volume  # decrease outstanding LO volume
                                self.X_t -= (
                                    mm_trade_volume * absolute_level
                                )  # deduct cash
                                self.Q_t += mm_trade_volume  # adjust inventory

                        else:
                            # IMPLICIT ASSUMPTION THAT MARKET MAKER HAS FIRST ORDER PRIORITY
                            mm_trade_volume = np.min([size, self.order_volumes["bid"]])
                            self.order_volumes["bid"] -= mm_trade_volume
                            self.X_t -= mm_trade_volume * absolute_level
                            self.Q_t += mm_trade_volume

                        vol_on_mm_bid -= size

                    # event type 3: 'mo ask', i.e., MO buy order arrives
                    elif (
                        event == 3
                        and absolute_level == self.quotes_absolute_level["ask"]
                    ):
                        if not self.mm_priority:
                            # IMPLICIT ASSUMPTION THAT MARKET MAKER HAS LAST ORDER PRIORITY
                            if vol_on_mm_ask - size < self.order_volumes["ask"]:
                                mm_trade_volume = size - (
                                    vol_on_mm_ask - self.order_volumes["ask"]
                                )
                                self.order_volumes[
                                    "ask"
                                ] -= mm_trade_volume  # decrease outstanding LO volume
                                self.X_t += (
                                    mm_trade_volume * absolute_level
                                )  # increase cash
                                self.Q_t -= mm_trade_volume  # adjust inventory

                        else:
                            # IMPLICIT ASSUMPTION THAT MARKET MAKER HAS FIRST ORDER PRIORITY
                            mm_trade_volume = np.min([size, self.order_volumes["ask"]])
                            self.order_volumes["ask"] -= mm_trade_volume
                            self.X_t += mm_trade_volume * absolute_level
                            self.Q_t -= mm_trade_volume

                        vol_on_mm_ask -= size

        if self.debug:
            if not self.MO_action:
                if (
                    self.X_t
                    != self.print_MO_results(
                        simulation_results, self.X_t_previous, printing=False
                    )
                    and self.t < self.T
                ):
                    self.num_errors += 1
                    self.print_MO_results(
                        simulation_results, self.X_t_previous, printing=True
                    )
                    print("--> should have been", self.X_t, "at time", self.t)
                    input("Next:")

        # If the market maker has an outstanding buy order, cancel the entire volume
        if self.t != self.T:
            if self.order_volumes["bid"] > 0:
                self.mc_model.ob.change_volume(
                    level=self.quotes_absolute_level["bid"],
                    volume=np.min(
                        [
                            self.order_volumes["bid"],
                            -self.mc_model.ob.get_volume(
                                self.quotes_absolute_level["bid"], absolute_level=True
                            ),
                        ]
                    ),
                    absolute_level=True,
                )

            # If the market maker has an outstanding sell order, cancel the entire volume
            if self.order_volumes["ask"] > 0:
                self.mc_model.ob.change_volume(
                    level=self.quotes_absolute_level["ask"],
                    volume=-np.min(
                        [
                            self.order_volumes["ask"],
                            self.mc_model.ob.get_volume(
                                self.quotes_absolute_level["ask"], absolute_level=True
                            ),
                        ]
                    ),
                    absolute_level=True,
                )

        if self.Q_t > 0:
            self.H_t = self.mc_model.ob.sell_n(self.Q_t)
        else:
            self.H_t = -self.mc_model.ob.buy_n(-self.Q_t)

        return self.state(), self._get_reward()

    def _get_reward(self):
        """
        Returns the reward

        Parameters
        ----------
        None

        Returns
        -------
        reward : float
            the reward
        """

        return (
            self.X_t
            + self.H_t
            - (self.X_t_previous + self.H_t_previous)
            - self.phi * self.Q_t**2
        )

    def pre_run(self, n_steps=int(1e4)):
        """
        Simulates n_steps events / transitions in the Markov chain LOB

        Parameters
        ----------
        n_steps : int
            the number of events that should be simulated in the LOB

        Returns
        -------
        None
        """

        self.mc_model.simulate(int(n_steps))

    def reset(self, randomized=None):
        """
        Reset the environment. Chooses a random LOB state if prompted.

        Parameters
        ----------
        randmoized : bool
            if the LOB state should be randomized

        Returns
        -------
        None
        """

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

        self.mc_model = MarkovChainLobModel(
            include_spread_levels=self.include_spread_levels,
            ob_start=ob_start,
            num_levels=self.num_levels,
        )
        if self.pre_run_on_start:
            self.pre_run()

    def render(self, mode="human"):
        """
        Prints useful stats of the environment

        Parameters
        ----------
        mode : str
            ??

        Returns
        -------
        None
        """

        print("=" * 40)
        print(f"End of t = {self.t}")
        print(f"Current bid-ask = {self.mc_model.ob.bid}-{self.mc_model.ob.ask}")
        print(f"Current inventory = {self.Q_t}")
        print(f"Cash value = {self.X_t}")
        print(f"Holding value = {self.H_t}")
        print(f"Total value = {self.X_t + self.H_t}")
        print(f"State = {self.state()}")
        print("=" * 40 + "\n")

    def print_simulation_results(self, simulation_results):
        """
        Prints the events after a simulation

        Parameters
        ----------
        simulation_results : dict
            a dictionary with events

        Returns
        -------
        None
        """

        for key in simulation_results.keys():
            if key == "event":
                print("\n event : ", end="")
                for event in simulation_results[key]:
                    print(self.mc_model.inverse_event_types[event], end=", ")
            else:
                print("\n", key, ":", simulation_results[key], end="")

    def print_MO_results(self, simulation_results, cash, printing=True):
        """
        Prints the MOs after a simulation. Used for debugging.

        Parameters
        ----------
        simulation_results : dict
            a dictionary with events

        Returns
        -------
        None
        """

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
                if (
                    self.mc_model.inverse_event_types[event] == "mo bid"
                    and level == self.quotes_absolute_level["bid"]
                ):
                    if printing:
                        print("MO sell, size", str(size_bid_event) + ", price", level)
                        print(
                            "cash:",
                            cash,
                            "-->",
                            cash,
                            "-",
                            size_bid_event,
                            "*",
                            level,
                            "=",
                            cash - size_bid_event * level,
                        )
                    cash -= size_bid_event * level
                    size_bid -= size_bid_event
                elif (
                    self.mc_model.inverse_event_types[event] == "mo ask"
                    and level == self.quotes_absolute_level["ask"]
                ):
                    if printing:
                        print("MO buy, size", str(size_ask_event) + ", price", level)
                        print(
                            "cash:",
                            cash,
                            "-->",
                            cash,
                            "+",
                            size_ask_event,
                            "*",
                            level,
                            "=",
                            cash + size_ask_event * level,
                        )
                    cash += size_ask_event * level
                    size_ask -= size_ask_event
            if printing:
                print("=" * 40)
        return cash


def init_LOB(num_levels=10, init_ask=10000, init_spread=1, init_volume=1):
    """
    Initiates a neutral LOB state and returns it

    Parameters
    ----------
    num_levels : int
        how many bid and ask levels there should be
    init_ask : int
        the starting ask price
    init_spread : int
        the starting spread
    init_volume : int
        the starting volume at all depths

    Returns
    -------
    lob : np.array
        the LOB state
    """

    lob = np.zeros((2, num_levels + 1), dtype=int)
    lob[0, 0] = init_ask
    lob[1, 0] = -init_spread
    lob[0, init_spread:] = init_volume
    lob[1, init_spread:] = -init_volume
    return lob
