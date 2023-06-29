import numpy as np
from environments.mc_model.lob_utils.lob_functions import *

"""
================== CREDITS =======================
The code in this file was written by Hanna Hultin.

"""


class MarkovChainLobModel:
    def __init__(
        self, num_levels, rates=None, ob_start=None, include_spread_levels=True
    ):
        """
        Parameters
        ----------
        num_levels : int
            number of levels of the order book
        rates : dict
            rates for different events
        ob_start : numpy array
            starting lob
        include_spread_levels : bool
            whether to include the spread levels in the LOB arrays

        Returns
        -------
        None
        """

        self.num_levels = num_levels
        self.event_types = {
            "lo buy": 0,
            "lo sell": 1,
            "mo bid": 2,
            "mo ask": 3,
            "cancellation buy": 4,
            "cancellation sell": 5,
        }
        self.inverse_event_types = {v: k for k, v in self.event_types.items()}

        self.next_event = None

        self.order_volumes = {"bid": 0, "ask": 0}
        self.order_prices = {"bid": None, "ask": None}

        if rates is None:
            self.rates = self.all_rates()
        else:
            self.rates = rates

        if ob_start is None:
            self.ob = LOB(
                np.zeros((2, self.num_levels + 1), dtype=int),
                include_spread_levels=include_spread_levels,
            )
        else:
            self.ob = LOB(
                ob_start.astype(int), include_spread_levels=include_spread_levels
            )

        self.current_rates = self.update_current_rates()
        return

    @staticmethod
    def geom_dist_probs(alpha, k=10):
        """
        Computes k first probabilities for discrete exponential with parameter alpha

        Parameters
        ----------
        alpha : float
            parameter for the distribution to sample from
        k : int
            number of probabilities

        Returns
        -------
        list of k first probabilities
        """
        return (np.exp(alpha) - 1) * np.exp(-alpha * np.arange(1, k + 1))

    def all_rates(self):
        """
        Compute all rates of the different processes.

        Returns
        -------
        rates : dict
            all rates
        """
        rates = {"mo bid": 0.0467}
        rates["mo ask"] = rates["mo bid"]

        lo_rates = (
            np.array([0.1330, 0.1811, 0.2085, 0.1477, 0.0541])
            + np.array([0.1442, 0.1734, 0.2404, 0.1391, 0.0584])
        ) / 2
        # lo_rates = [.1330, .1811, .2085, .1477, .0541]
        rates["lo buy"] = np.zeros(self.num_levels)
        rates["lo buy"][: min(len(lo_rates), self.num_levels)] = lo_rates[
            : self.num_levels
        ]

        # lo_rates = [.1442, .1734, .2404, .1391, .0584]
        rates["lo sell"] = np.zeros(self.num_levels)
        rates["lo sell"][: min(len(lo_rates), self.num_levels)] = lo_rates[
            : self.num_levels
        ]

        rates["cancellation buy"] = np.zeros(self.num_levels)
        canc_rates = (
            np.array([0.1287, 0.1057, 0.0541, 0.0493, 0.0408])
            + np.array([0.1308, 0.1154, 0.0531, 0.0492, 0.0437])
        ) / 2
        # canc_rates = [.1287, .1057, .0541, .0493, .0408]
        rates["cancellation buy"][: min(len(canc_rates), self.num_levels)] = canc_rates[
            : self.num_levels
        ]
        rates["cancellation sell"] = np.zeros(self.num_levels)
        # canc_rates = [.1308, .1154, .0531, .0492, .0437]
        rates["cancellation sell"][
            : min(len(canc_rates), self.num_levels)
        ] = canc_rates[: self.num_levels]

        rates["lo size"] = 0.5667
        rates["mo size"] = 0.4955

        rates["lo probs"] = self.geom_dist_probs(rates["lo size"], k=self.num_levels)
        rates["mo probs"] = self.geom_dist_probs(rates["mo size"], k=self.num_levels)

        for i in range(self.num_levels):
            for s in ["lo buy", "lo sell", "cancellation buy", "cancellation sell"]:
                rates["{} {}".format(s, i)] = rates[s][i]
        return rates

    def update_current_rates(self):
        """
        Take out the rates for all events that could happen given current state of the order book

        Returns
        -------
            dict of all relevant rates
        """

        self.current_rates = {}
        if self.ob.ask_volume > 0:
            self.current_rates["mo ask"] = self.rates["mo ask"]
        if self.ob.bid_volume > 0:
            self.current_rates["mo bid"] = self.rates["mo bid"]

        for i in range(self.num_levels):
            self.current_rates["lo buy {}".format(i)] = self.rates[
                "lo buy {}".format(i)
            ]
            self.current_rates["lo sell {}".format(i)] = self.rates[
                "lo sell {}".format(i)
            ]

            if self.ob.get_volume(-i - 1) != 0:
                self.current_rates["cancellation buy {}".format(i)] = (
                    np.abs(self.ob.get_volume(-i - 1))
                    * self.rates["cancellation buy {}".format(i)]
                )
            if self.ob.get_volume(self.ob.relative_bid + i + 1) != 0:
                self.current_rates["cancellation sell {}".format(i)] = (
                    np.abs(self.ob.get_volume(self.ob.relative_bid + i + 1))
                    * self.rates["cancellation sell {}".format(i)]
                )
        return self.current_rates

    def update_current_rates_new(self):
        """
        Updates the current rate

        Returns
        -------
        the updated rates
        """

        self.current_rates = {}
        if self.ob.ask_volume > 0:
            self.current_rates["mo ask"] = self.rates["mo ask"]
        if self.ob.bid_volume > 0:
            self.current_rates["mo bid"] = self.rates["mo bid"]

        for i in range(self.num_levels):
            self.current_rates["lo buy {}".format(i)] = self.rates[
                "lo buy {}".format(i)
            ]
            self.current_rates["lo sell {}".format(i)] = self.rates[
                "lo sell {}".format(i)
            ]

            if self.ob.get_volume(-i - 1) != 0:
                if self.ob.bid_n(i, absolute_level=True) == self.order_prices["bid"]:
                    # volume on the mm's level that is not the market maker's
                    other_vol_on_level = (
                        np.abs(self.ob.get_volume(-i - 1)) - self.order_volumes["bid"]
                    )
                    self.current_rates["cancellation buy {}".format(i)] = (
                        other_vol_on_level * self.rates["cancellation buy {}".format(i)]
                    )
                else:
                    self.current_rates["cancellation buy {}".format(i)] = (
                        np.abs(self.ob.get_volume(-i - 1))
                        * self.rates["cancellation buy {}".format(i)]
                    )

            if self.ob.get_volume(self.ob.relative_bid + i + 1) != 0:
                if self.ob.ask_n(i, absolute_level=True) == self.order_prices["ask"]:
                    other_vol_on_level = (
                        np.abs(self.ob.get_volume(self.ob.relative_bid + i + 1))
                        - self.order_volumes["ask"]
                    )
                    self.current_rates["cancellation sell {}".format(i)] = (
                        other_vol_on_level
                        * self.rates["cancellation sell {}".format(i)]
                    )
                else:
                    self.current_rates["cancellation sell {}".format(i)] = (
                        np.abs(self.ob.get_volume(self.ob.relative_bid + i + 1))
                        * self.rates["cancellation sell {}".format(i)]
                    )
        return self.current_rates

    def sample_next_event(self):
        """
        Sample next event given rates, but without performing it and changing the current order book

        Returns
        -------
        the new order book after the transition
        """
        self.next_event = {}
        self.update_current_rates_new()
        with np.errstate(divide="ignore"):
            ts = np.random.exponential(1 / np.array(list(self.current_rates.values())))
        ri = np.argmin(ts)
        self.next_event["time"] = ts[ri]
        k = list(self.current_rates.keys())[ri]
        for (
            ke,
            ve,
        ) in self.event_types.items():
            if k.startswith(ke):
                self.next_event["event"] = ve
        self.next_event["size"] = 1

        if k.startswith("mo"):
            self.next_event["abs_level"] = self.ob.bid if k == "mo bid" else self.ob.ask
            self.next_event["level"] = 0
            ob_volume = self.ob.ask_volume if k == "mo ask" else self.ob.bid_volume
            if ob_volume <= 0:
                print(self.ob.data)
                print(self.next_event)
                print(self.current_rates)
            self.next_event["size"] = self.mo_size(self.rates["mo size"], ob_volume)

            if (
                k == "mo bid"
                and self.next_event["abs_level"] == self.order_prices["bid"]
            ):
                self.order_volumes["bid"] -= np.min(
                    [self.next_event["size"], self.order_volumes["bid"]]
                )
            elif (
                k == "mo ask"
                and self.next_event["abs_level"] == self.order_prices["ask"]
            ):
                self.order_volumes["ask"] -= np.min(
                    [self.next_event["size"], self.order_volumes["ask"]]
                )

        elif k.startswith("lo"):
            i = int(k.rpartition(" ")[-1])
            self.next_event["level"] = i
            self.next_event["size"] = np.random.geometric(
                p=1 - np.exp(-self.rates["lo size"])
            )

            if k.startswith("lo buy"):
                self.next_event["abs_level"] = self.ob.ask - i - 1
            else:
                self.next_event["abs_level"] = self.ob.bid + i + 1
        else:
            i = int(k.rpartition(" ")[-1])
            self.next_event["level"] = i
            self.next_event["abs_level"] = (
                self.ob.ask - i - 1
                if k.startswith("cancellation buy")
                else self.ob.bid + i + 1
            )

        return self.next_event

    def transition(self, event_dict=None):
        """
        Transition the order book according to a given event, if no event is given, samples an event given rates

        Parameters
        ----------
        event_dict : dictionary
            describing next event with
                "event" (int)
                "level" (int)
                "abs_level" (int)
                "size" (int)
                "time" (float)

        Returns
        -------
        the new order book after the transition
        """
        if event_dict is None:
            event_dict = (
                self.sample_next_event()
                if self.next_event is None
                else self.next_event.copy()
            )

        if self.inverse_event_types[event_dict["event"]] in [
            "mo bid",
            "lo sell",
            "cancellation buy",
        ]:
            change_ok = self.ob.change_volume(
                event_dict["abs_level"], event_dict["size"], absolute_level=True
            )
        else:
            change_ok = self.ob.change_volume(
                event_dict["abs_level"], -event_dict["size"], absolute_level=True
            )
        event_dict["ob"] = self.ob.data.copy()
        self.next_event = None
        return event_dict, change_ok

    @staticmethod
    def mo_size(alpha, k):
        """
        Sample the size of a market order

        Parameters
        ----------
        alpha : float
            parameter for the distribution to sample from
        k : int
            current available volume at the touch

        Returns
        -------
        s
            int of size of MO
        """
        p = 1 - np.exp(-alpha)
        s = 0
        while s == 0:
            s = np.random.geometric(p=p)
            if s > k:
                s = 0
        return s

    def simulate(
        self,
        num_events=int(1e3),
        end_time=np.inf,
        num_print=np.inf,
        order_volumes=None,
        order_prices=None,
    ):
        """
        Simulate the order book for a number of events

        Parameters
        ----------
        num_events : int
            number of events
        end_time : float
            max time to simulate
        num_print : int
            how ofter to print current event number

        Returns
        -------
        data_dict
            dictionary containing data about the simulation including
            ob: numpy array of shape (num_events+1, n) containing the LOB in all timesteps
            event: numpy array of event types in all timesteps
            level: numpy array of levels in all timesteps
            abs_level: numpy array of absolute levels in all timesteps
            time: numpy array of time between all events
        """

        data_dict = {
            "ob": np.zeros((num_events + 1, 2, self.num_levels + 1), dtype=int)
        }
        data_dict["ob"][0, ...] = self.ob.data
        data_dict["time"] = np.zeros(num_events)

        if order_volumes is not None and order_prices is not None:
            self.order_volumes = order_volumes.copy()
            self.order_prices = order_prices.copy()
        else:
            self.order_volumes = {"bid": 0, "ask": 0}
            self.order_prices = {"bid": None, "ask": None}

        for k in ["event", "level", "size", "index", "abs_level"]:
            data_dict[k] = np.zeros(num_events, dtype=int)

        data_dict["total_time"] = 0
        for e in range(num_events):
            if (e + 1) % num_print == 0:
                print("event: ", e)

            event_dict = self.sample_next_event()

            if data_dict["total_time"] + event_dict["time"] < end_time:
                data_dict["total_time"] += event_dict["time"]
                self.transition(event_dict)
                for k, v in event_dict.items():
                    if k != "ob":
                        data_dict[k][e] = v
                data_dict["ob"][e + 1, ...] = self.ob.data

            else:
                data_dict["total_time"] = end_time
                data_dict["ob"] = data_dict["ob"][: e + 1, ...]
                for k in ["event", "level", "size", "index", "time", "abs_level"]:
                    data_dict[k] = data_dict[k][:e]
                e -= 1
                break

        data_dict["num_events"] = e + 1
        return data_dict

    def simulate_old(self, num_events=int(1e3), end_time=np.inf, num_print=np.inf):
        """
        Simulate the order book for a number of events

        Parameters
        ----------
        num_events : int
            number of events
        end_time : float
            max time to simulate
        num_print : int
            how ofter to print current event number

        Returns
        -------
        data_dict
            dictionary containing data about the simulation including
            ob: numpy array of shape (num_events+1, n) containing the LOB in all timesteps
            event: numpy array of event types in all timesteps
            level: numpy array of levels in all timesteps
            abs_level: numpy array of absolute levels in all timesteps
            time: numpy array of time between all events
        """

        data_dict = {
            "ob": np.zeros((num_events + 1, 2, self.num_levels + 1), dtype=int)
        }
        data_dict["ob"][0, ...] = self.ob.data
        data_dict["time"] = np.zeros(num_events)

        for k in ["event", "level", "size", "index", "abs_level"]:
            data_dict[k] = np.zeros(num_events, dtype=int)

        data_dict["total_time"] = 0
        for e in range(num_events):
            if (e + 1) % num_print == 0:
                print("event: ", e)

            event_dict = self.sample_next_event()

            if data_dict["total_time"] + event_dict["time"] < end_time:
                data_dict["total_time"] += event_dict["time"]
                self.transition(event_dict)
                for k, v in event_dict.items():
                    if k != "ob":
                        data_dict[k][e] = v
                data_dict["ob"][e + 1, ...] = self.ob.data

            else:
                data_dict["total_time"] = end_time
                data_dict["ob"] = data_dict["ob"][: e + 1, ...]
                for k in ["event", "level", "size", "index", "time", "abs_level"]:
                    data_dict[k] = data_dict[k][:e]
                e -= 1
                break

        data_dict["num_events"] = e + 1
        return data_dict

    def run_twap(
        self, volume, end_time, timefactor=100, initial_ob=None, padding=(0, 0)
    ):
        """
        Runs a TWAP buying the volume with timedelta between each buy.

        Parameters
        ----------
        volume : int
            total volume to buy
        end_time : float
            total time to simulate
        timefactor : float
            how often to save mid/ask/bid of order book
        initial_ob : numpy array
            starting point of the LOB
        padding : tuple of two floats
            how much time to add before the first trade/after the last

        Returns
        -------
        twap_dict
            dictionary containing information about the twap
        """
        if type(initial_ob) is dict:
            initial_ob = initial_ob["ob"]
        if initial_ob is not None:
            self.ob = LOB(initial_ob.copy(), self.ob.include_spread_levels)

        total_time = 0
        twap_dict = {}

        if volume > 1:
            timedelta = (end_time - padding[0] - padding[1]) / (volume - 1)
        else:
            timedelta = 1

        num_times = int(end_time * timefactor) + 2
        for s in ["mid_vals", "ask_vals", "bid_vals"]:
            twap_dict[s] = np.zeros(num_times)
        twap_dict["mid_vals"][0] = self.ob.mid
        twap_dict["ask_vals"][0] = self.ob.ask
        twap_dict["bid_vals"][0] = self.ob.bid
        twap_dict["time_vals"] = np.zeros(num_times + 1)
        twap_dict["time_vals"][:-1] = np.linspace(
            -(1 / timefactor), end_time, num_times
        )
        twap_dict["time_vals"][-1] = np.inf
        time_index = 1

        twap_dict["num_events"] = 0
        twap_dict["num_mo"] = 0

        mid_current = self.ob.mid
        ask_current = self.ob.ask
        bid_current = self.ob.bid

        while total_time < padding[0]:
            twap_dict["num_events"] += 1
            event_dict = self.sample_next_event()
            total_time += event_dict["time"]
            if total_time < padding[0]:
                self.transition()
                if self.inverse_event_types[event_dict["event"]] in [
                    "mo bid",
                    "mo ask",
                ]:
                    twap_dict["num_mo"] += 1
            else:
                total_time = padding[0]

            while total_time > twap_dict["time_vals"][time_index]:
                twap_dict["mid_vals"][time_index] = mid_current
                twap_dict["ask_vals"][time_index] = ask_current
                twap_dict["bid_vals"][time_index] = bid_current
                time_index += 1

            mid_current = self.ob.mid
            ask_current = self.ob.ask
            bid_current = self.ob.bid

        volume_left = volume - 1
        event_dict = {
            "event": self.event_types["mo ask"],
            "abs_level": self.ob.ask,
            "size": 1,
            "index": 1,
            "level": 0,
        }
        twap_dict["price"] = self.ob.ask
        self.transition(event_dict)

        mid_current = self.ob.mid
        ask_current = self.ob.ask
        bid_current = self.ob.bid

        if total_time == twap_dict["time_vals"][time_index]:
            twap_dict["mid_vals"][time_index] = mid_current
            twap_dict["ask_vals"][time_index] = ask_current
            twap_dict["bid_vals"][time_index] = bid_current
            time_index += 1

        for v in range(volume_left):
            while total_time < (padding[0] + timedelta * (v + 1)):
                twap_dict["num_events"] += 1
                event_dict = self.sample_next_event()
                total_time += event_dict["time"]
                if total_time < (padding[0] + timedelta * (v + 1)):
                    self.transition()
                    if self.inverse_event_types[event_dict["event"]] in [
                        "mo bid",
                        "mo ask",
                    ]:
                        twap_dict["num_mo"] += 1

                else:
                    total_time = padding[0] + timedelta * (v + 1)
                    twap_dict["price"] += self.ob.ask
                    event_dict = {
                        "event": self.event_types["mo ask"],
                        "abs_level": self.ob.ask,
                        "size": 1,
                        "level": 0,
                    }
                    self.transition(event_dict)

                while total_time > twap_dict["time_vals"][time_index]:
                    twap_dict["mid_vals"][time_index] = mid_current
                    twap_dict["ask_vals"][time_index] = ask_current
                    twap_dict["bid_vals"][time_index] = bid_current
                    time_index += 1

                mid_current = self.ob.mid
                ask_current = self.ob.ask
                bid_current = self.ob.bid

                if total_time == twap_dict["time_vals"][time_index]:
                    twap_dict["mid_vals"][time_index] = mid_current
                    twap_dict["ask_vals"][time_index] = ask_current
                    twap_dict["bid_vals"][time_index] = bid_current
                    time_index += 1

        while total_time < end_time:
            twap_dict["num_events"] += 1
            event_dict = self.sample_next_event()
            total_time += event_dict["time"]
            if total_time < end_time:
                self.transition(event_dict)
                if self.inverse_event_types[event_dict["event"]] in [
                    "mo bid",
                    "mo ask",
                ]:
                    twap_dict["num_mo"] += 1
            else:
                total_time = end_time

            while total_time > twap_dict["time_vals"][time_index]:
                twap_dict["mid_vals"][time_index] = mid_current
                twap_dict["ask_vals"][time_index] = ask_current
                twap_dict["bid_vals"][time_index] = bid_current
                time_index += 1

            mid_current = self.ob.mid
            ask_current = self.ob.ask
            bid_current = self.ob.bid

            if total_time == twap_dict["time_vals"][time_index]:
                twap_dict["mid_vals"][time_index] = mid_current
                twap_dict["ask_vals"][time_index] = ask_current
                twap_dict["bid_vals"][time_index] = bid_current
                time_index += 1

        twap_dict["finished"] = True
        return twap_dict
