import numpy as np

"""
================== CREDITS =======================
The code in this file was written by Hanna Hultin.

"""


class LOB:
    def __init__(self, data, outside_volume=1, include_spread_levels=True):
        self.data = data.reshape((2, -1))
        self.num_levels = self.data.shape[1] - 1
        self.outside_volume = outside_volume
        self.include_spread_levels = include_spread_levels

    @property
    def ask(self):
        return int(self.data[0, 0])

    @property
    def bid(self):
        return int(self.data[0, 0] + self.data[1, 0])

    @property
    def mid(self):
        return (self.bid + self.ask) / 2

    @property
    def vwap_mid(self):
        vwap_a = np.dot(
            self.data[0, 1:], self.bid + 1 + np.arange(self.num_levels)
        ) / np.sum(self.data[0, 1:])
        vwap_b = np.dot(
            -self.data[1, 1:], self.ask - 1 - np.arange(self.num_levels)
        ) / np.sum(-self.data[1, 1:])
        vwap = (vwap_a + vwap_b) / 2
        return vwap

    @property
    def microprice(self):
        bid_volume = self.bid_volume if self.bid_volume != 0 else self.outside_volume
        ask_volume = self.ask_volume if self.ask_volume != 0 else self.outside_volume
        if bid_volume == 0 and ask_volume == 0:
            return self.mid
        else:
            return (self.ask * bid_volume + self.bid * ask_volume) / (
                bid_volume + ask_volume
            )

    @property
    def spread(self):
        return int(-self.data[1, 0])

    @property
    def relative_bid(self):
        return int(self.data[1, 0])

    @property
    def ask_volume(self):
        return (
            0
            if self.spread > self.num_levels
            else (
                self.data[0, self.spread]
                if self.include_spread_levels
                else self.data[0, 1]
            )
        )

    @property
    def bid_volume(self):
        return (
            0
            if self.spread > self.num_levels
            else (
                -self.data[1, self.spread]
                if self.include_spread_levels
                else -self.data[1, 1]
            )
        )

    def buy_n(self, n=1):
        total_price = 0
        level = 0
        while n > 0:
            if level >= self.num_levels:
                available = self.outside_volume
                if self.outside_volume == 0:
                    print("TOO LARGE VOLUME")
                    return total_price
            else:
                available = self.ask_n_volume(level)
            vol = np.min((n, available))
            n -= vol
            total_price += (self.bid + 1 + level) * vol
            level += 1
        return total_price

    def sell_n(self, n=1):
        total_price = 0
        level = 0
        while n > 0:
            if level >= self.num_levels:
                available = self.outside_volume
                if self.outside_volume == 0:
                    print("TOO LARGE VOLUME")
                    return total_price
            else:
                available = self.bid_n_volume(level)
            vol = np.min((n, available))
            n -= vol
            total_price += (self.ask - 1 - level) * vol
            level += 1
        return total_price

    def ask_n(self, n=0, absolute_level=False):
        if n == 0 and not absolute_level:
            return 0
        level = int(self.relative_bid + n + 1) if self.include_spread_levels else n
        if absolute_level:
            level += self.ask
        return level

    def bid_n(self, n=0, absolute_level=False):
        level = -n - 1 if self.include_spread_levels else int(self.relative_bid - n)
        if absolute_level:
            level += self.ask
        return level

    def ask_n_volume(self, n=0):
        return self.data[0, n + 1]

    def bid_n_volume(self, n=0):
        return -self.data[1, n + 1]

    def q_ask(self, num_levels=None):
        if num_levels is None:
            num_levels = self.num_levels
        x = self.data[0, 1 : 1 + num_levels]
        return x

    def q_bid(self, num_levels=None):
        if num_levels is None:
            num_levels = self.num_levels
        x = -self.data[1, 1 : 1 + num_levels]
        return x

    def get_volume(self, level, absolute_level=False):
        if absolute_level:
            level = level - self.ask
        if level == 0:
            return self.ask_volume
        if self.include_spread_levels:
            if (level > 0) and ((self.spread + level - 1) < self.num_levels):
                return self.data[0, self.spread + level]
            elif (level < 0) and ((-level - 1) < self.num_levels):
                return self.data[1, -level]
            else:
                return 0
        else:
            if (level > 0) and (level < self.num_levels):
                return self.data[0, level + 1]
            elif (level < 0) and (-level < self.spread):
                return 0
            elif (level < 0) and (-level - self.spread < self.num_levels):
                return self.data[1, -level - self.spread + 1]
            else:
                return 0

    def order_imbalance(self, depth=None):
        """
        Compute order imbalance for levels up to a specified depth from best bid/ask
        such that high order imbalance means more volume available on bid side (more volume wanting to buy).

        Parameters
        ----------
        depth : int
            specifies how many levels from (and including) best bid and best ask to consider

        Returns
        -------
        float between 0 and 1 of order imbalance

        """
        if depth is None:
            depth = self.num_levels
        vol_sell = np.sum(
            self.data[0, self.spread : self.spread + depth]
            if self.include_spread_levels
            else self.data[0, 1 : 1 + depth]
        )
        vol_buy = np.sum(
            -self.data[1, self.spread : self.spread + depth]
            if self.include_spread_levels
            else -self.data[1, 1 : 1 + depth]
        )
        if vol_buy + vol_sell == 0:
            return 0.5
        else:
            return (vol_buy - vol_sell) / (vol_buy + vol_sell)

    def change_volume(self, level, volume, absolute_level=False, print_info=False):
        if absolute_level:
            level -= self.ask

        offset = self.spread if self.include_spread_levels else 1
        if level == 0:
            if self.ask_volume + volume < -1e-6:
                if print_info:
                    print("LEVEL 0")
                    print(self.data)
                    print("level: ", level)
                    print("volume: ", volume)
                    print(self.ask_volume + volume)
                return False
            self.data[0, np.min([offset, self.num_levels])] = np.round(
                self.data[0, np.min([offset, self.num_levels])] + volume, decimals=6
            )

            # if ask volume 0, move ask
            if self.data[0, offset] == 0:
                old_spread = self.spread
                index = np.argwhere(self.data[0, offset:])
                if index.size == 0:
                    if print_info:
                        print(self.data)
                        print(level)
                        print(volume)
                    move_ask = self.num_levels - offset + 1
                    self.data[0, 0] += move_ask
                    self.data[1, 0] -= move_ask
                    self.data[:, 1:] = 0
                else:
                    move_ask = index.flat[0]
                    self.data[0, 0] += move_ask
                    self.data[1, 0] -= move_ask
                    if self.include_spread_levels:
                        self.data[0, old_spread] = 0
                        self.data[1, 1 + move_ask :] = self.data[1, 1:-move_ask]
                        self.data[1, 1 : 1 + move_ask] = 0
                    else:
                        self.data[0, 1:-move_ask] = self.data[0, 1 + move_ask :]
                        self.data[0, -move_ask:] = self.outside_volume
            return True
        elif level > 0:
            # outside of LOB range
            if level > (self.num_levels - offset):
                return True
            # taking away too much volume not possible
            elif self.data[0, offset + level] + volume < 0:
                return False
            # transition by adding volume on ask side
            else:
                self.data[0, offset + level] = np.round(
                    self.data[0, offset + level] + volume, decimals=6
                )
                return True

        else:
            # outside of LOB range
            if -level > self.num_levels + self.spread - offset:
                return True

            level_index = (
                -level if self.include_spread_levels else -level - self.spread + 1
            )

            # if adding volume inside the spread
            if self.spread + level > 0:
                old_spread = self.spread

                # new bid
                if volume < 0:
                    self.data[1, 0] = level
                    if self.include_spread_levels:
                        self.data[1, -level] = volume
                        self.data[0, 1 : -old_spread - level] = self.data[
                            0, 1 + old_spread + level :
                        ]
                        self.data[0, -old_spread - level :] = self.outside_volume
                    else:
                        self.data[1, 1 + old_spread + level :] = self.data[
                            1, 1 : -(old_spread + level)
                        ]
                        self.data[1, 1 : old_spread + level] = 0
                        self.data[1, old_spread + level] = volume
                    return True

                # new ask
                else:
                    old_ask = self.ask
                    self.data[0, 0] = old_ask + level
                    self.data[1, 0] -= level
                    if self.include_spread_levels:
                        self.data[0, self.spread] = volume
                        self.data[1, 1:level] = self.data[1, 1 - level :]
                        self.data[1, level:] = -self.outside_volume
                    else:
                        self.data[0, 1 + old_spread + level :] = self.data[
                            0, 1 : -(old_spread + level)
                        ]
                        self.data[0, 1 : old_spread + level] = 0
                        self.data[0, old_spread + level] = volume
                    return True

            # taking away too much volume not possible
            elif self.data[1, level_index] + volume > 0:
                return False

            # transition by adding volume on bid side
            else:
                self.data[1, level_index] = np.round(
                    self.data[1, level_index] + volume, decimals=6
                )

                # adjust if best bid changed
                if (self.data[1, level_index] == 0) and (self.relative_bid == level):
                    old_spread = self.spread
                    index = np.argwhere(self.data[1, level_index + 1 :])
                    if index.size == 0:
                        if print_info:
                            print(self.data)
                            print(level)
                            print(volume)
                        move_bid = self.num_levels - offset + 1
                        self.data[1, 0] -= move_bid
                        self.data[:, 1:] = 0
                    else:
                        move_bid = index.flat[0] + 1
                        self.data[1, 0] -= move_bid
                        if self.include_spread_levels:
                            self.data[1, old_spread] = 0
                            self.data[0, 1 + move_bid :] = self.data[0, 1:-move_bid]
                            self.data[0, 1 : 1 + move_bid] = 0
                        else:
                            self.data[1, 1:-move_bid] = self.data[1, 1 + move_bid :]
                            self.data[1, -move_bid] = -self.outside_volume
                return True


def ask(x):
    """
    Compute best ask of limit order book

    Parameters
    ----------
    x : numpy array
        volumes of each level of the order book

    Returns
    -------
    int of ask price

    """
    pa = np.where(x > 0)[0]
    if len(pa) == 0:
        p = len(x)
    else:
        p = pa[0]
    return p


def bid(x):
    """
    Compute best bid of limit order book

    Parameters
    ----------
    x : numpy array
        volumes of each level of the order book

    Returns
    -------
    int of bid price

    """
    pa = np.where(x < 0)[0]
    if len(pa) == 0:
        p = -1
    else:
        p = pa[-1]
    return p


def mid(x):
    """
    Compute mid of limit order book using best bid and best ask

    Parameters
    ----------
    x : numpy array
        volumes of each level of the order book

    Returns
    -------
    int or float of mid price

    """
    return (ask(x) + bid(x)) / 2


def spread(x):
    """
    Compute spread of limit order book

    Parameters
    ----------
    x : numpy array
        volumes of each level of the order book

    Returns
    -------
    int of spread

    """
    return ask(x) - bid(x)


def q_bid(x):
    """
    Computes volumes on bid side by distance from best ask price

    Parameters
    ----------
    x : numpy array
        volumes of each level of the order book

    Returns
    -------
    numpy array with element i containing the volume at distance i from ask

    """
    q = np.zeros(len(x))
    p = ask(x)
    q[:p] = np.flip(x[:p])
    return q


def q_ask(x):
    """
    Computes volumes on ask side by distance from best bid price

    Parameters
    ----------
    x : numpy array
        volumes of each level of the order book

    Returns
    -------
    numpy array with element i containing the volume at distance i from bid

    """
    q = np.zeros(len(x))
    p = bid(x)
    if p >= 0:
        q[: -(p + 1)] = x[p + 1 :]
    else:
        q = x
    return q


def vwap_mid(x):
    """
    Compute volume-weighted average mid of limit order book

    Parameters
    ----------
    x : numpy array
        volumes of each level of the order book

    Returns
    -------
    int or float of vwap mid price

    """
    prices = np.arange(len(x))
    b = bid(x)
    a = ask(x)
    vwap_b = np.dot(x[: b + 1], prices[: b + 1]) / np.sum(x[: b + 1])
    vwap_a = np.dot(x[a:], prices[a:]) / np.sum(x[a:])
    vwap = (vwap_a + vwap_b) / 2
    return vwap


def buy_volume(x, depth):
    """
    Compute volume available on bid size wanting to buy up to specified depth from best bid

    Parameters
    ----------
    x : numpy array
        volumes of each level of the order book
    depth : int
        specifies how many levels from (and including) best bid to consider

    Returns
    -------
    int of volume

    """
    return -np.sum(x[np.maximum(bid(x) - depth, 0) : bid(x) + 1])


def sell_volume(x, depth):
    """
    Compute volume available on ask size wanting to sell up to specified depth from best ask

    Parameters
    ----------
    x : numpy array
        volumes of each level of the order book
    depth : int
        specifies how many levels from (and including) best ask to consider

    Returns
    -------
    int of volume

    """
    return np.sum(x[ask(x) : ask(x) + 1 + depth])


def realized_vol(prices, seq_length=-1):
    """
    Compute realized volatility for each sequence of prices

    Parameters
    ----------
    prices : numpy array
        prices (mid/ask/bid/other) of shape (n,m) with n sequences each of m observations
    seq_length : int
        how long each subsequence should be, if -1 then each sequence considered as one

    Returns
    -------
    numpy array of shape (n,) of realized volatilities for each sequence

    """
    if seq_length != -1:
        num_seq = int(prices.shape[1] / seq_length)
        price_seqs = np.zeros((prices.shape[0], num_seq, seq_length + 1))
        for n in range(num_seq):
            price_seqs[:, n, :] = prices[:, n * seq_length : (n + 1) * seq_length + 1]
        prices = price_seqs.reshape((-1, seq_length + 1))

    return np.std(np.diff(prices, axis=-1), axis=-1)


def order_imbalance(x, depth=None):
    """
    Compute order imbalance for levels up to a specified depth from best bid/ask
    such that high order imbalance means more volume available on bid side (more volume wanting to buy).

    Parameters
    ----------
    x : numpy array
        volumes of each level of the order book
    depth : int
        specifies how many levels from (and including) best bid and best ask to consider

    Returns
    -------
    float between 0 and 1 of order imbalance

    """
    if not depth:
        depth = len(x)
    vol_buy = buy_volume(x, depth)
    vol_sell = sell_volume(x, depth)
    return (vol_buy - vol_sell) / (vol_buy + vol_sell)


def price_signature(p, p0, sizes=None):
    """
    Compute price signature for prices p given reference price p0.

    Parameters
    ----------
    p : numpy array of shape (m,n,d)
        here n is the number of price series for each type of sequence and m is the number of types of such sequences
        d is the number of prices for each price series
    p0 : numpy array of shape (m,n)
        reference price for each price series
    sizes : numpy array of (m,n)
        sizes of the orders to normalize the signature by

    Returns
    -------
    s : numpy array of shape (m, d)

    """
    n = p.shape[1]
    d = p.shape[2]

    p0 = np.tile(p0[:, :, np.newaxis], d)
    if sizes is not None:
        sizes = np.tile(sizes[:, :, np.newaxis], d)
    s = (
        np.sum((p - p0) / sizes, axis=1) / n
        if sizes is not None
        else np.sum(p - p0, axis=1) / n
    )
    return s


def get_volumes(ob, num_levels=10, relative=True, negative_bids=False):
    """
    Compute volume feature vector from ob
    x = (p_ask^1, v_ask^1, p_bid^1, v_bid^1, ...., p_ask^n, v_ask^n, p_bid^n, v_bid^n)
    with p_ask^i being the price level of i:th best ask price (same for bid)
    and v_ask^i the volume of the i:th best ask price (same for bid)

    Parameters
    ----------
    ob : numpy array
        state of the order book
    num_levels : int
        number of levels to consider in feature vector for buy/sell side
    relative : bool
        whether to have all price levels relative to the first ask price or not
    negative_bids : bool
        whether the volumes should be negative on the bid size

    Returns
    -------
    x : numpy array of length 4 * num_levels with extracted features

    """

    x = np.zeros(4 * num_levels)
    x[0] = ask(ob)

    ask_nonzero = ob > 0
    ask_idx = np.argwhere(ask_nonzero)[:num_levels].flatten()
    x[4 : (4 * ask_idx.size) : 4] = ask_idx[1:] - x[0] if relative else ask_idx[1:]
    x[1 : (4 * ask_idx.size + 1) : 4] = ob[ask_idx]

    bid_nonzero = ob < 0
    bid_idx = np.flip(np.argwhere(bid_nonzero).flatten())[:num_levels]
    x[2 : (4 * bid_idx.size + 2) : 4] = bid_idx - x[0] if relative else bid_idx
    x[3 : (4 * bid_idx.size + 3) : 4] = (
        ob[bid_idx] if negative_bids else np.abs(ob[bid_idx])
    )

    return x
