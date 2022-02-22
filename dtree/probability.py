import itertools
import pandas as pd
from copy import deepcopy


def get_probas(name, config):
    """Returns the probability distribution for name, that is specified in config.

    Params:
        - name {str}: string of category for which we want the distribution
        - config {dict}: dictionary containing the changable parameters of the model

    Returns:
        - res {dict}: dictionary in the form
        {
            "p" : # dictionary with categorical distribution
        }

        for the particular item
    """
    p = {}
    for k, v in config.items():
        if (name in k) and "__" in k:
            p[k.split("__")[1]] = v
    return p


def coin_base(coin, p_stock, config):
    """
    Returns the probabilities of return levels for a coin.

    Params:
        - coin {str}: name of coin ["BTC", "ETH", "SOL"]
        - stockmarket_proba {dict}: probabilities of stockmarket return levels
        - config {dict}: dictionary containing the changable parameters of the model

    Returns:
        - BASE {dict}: dictionary in the form
        {
            "name": #name of the random variable
            "p" : # dictionary with categorical distribution
        }

        for the particular coin
    """
    mag1 = config[f"{coin}__mag1"]
    magx = 1 - mag1

    p_coin = dict()
    p_coin["low_low"] = magx * p_stock["low"]
    p_coin["low"] = mag1 * p_stock["low"] + 0.5 * magx * p_stock["neutral"]
    p_coin["neutral"] = mag1 * p_stock["neutral"]
    p_coin["high"] = mag1 * p_stock["high"] + 0.5 * magx * p_stock["neutral"]
    p_coin["high_high"] = magx * p_stock["high"]

    return p_coin


def market_adaption_cond_probas(weighting):
    """
    Takes weighting for each source of uncertainty for market adoption.
    Calculates the conditional probabilities and returns information in a
    DataFrame.

    Params:
        - weighting {Tuple}: tuple with weighting for
                            (public_perception, reg, technology)
    Returns:
        - df {pd.DataFrame}: dataframe with first 3 columns decribing the state
                            (reg positive/neutral/neg etc.)
                            Last column is market_adoption_cond_proba contains
                            tuple with conditional probabilities for marketadoption being
                            (low, neutral, high)
    """

    df = pd.DataFrame()
    s = pd.Series(list(itertools.product([0, 1, 2], [0, 1, 2], [0, 1, 2])))
    df["public_perception"] = s.apply(lambda x: x[0])
    df["reg"] = s.apply(lambda x: x[1])
    df["technology"] = s.apply(lambda x: x[2])

    def get_market_adoption_probability(row):
        negative = (row == 0).tolist()
        neutral = (row == 1).tolist()
        favorable = (row == 2).tolist()

        ma_low_p = sum([x * y for x, y in zip(negative, weighting)])
        ma_medium_p = sum([x * y for x, y in zip(neutral, weighting)])
        ma_high_p = sum([x * y for x, y in zip(favorable, weighting)])

        return (ma_low_p, ma_medium_p, ma_high_p)

    df["market_adoption_cond_proba"] = df.apply(get_market_adoption_probability, axis=1)
    return df


def market_adaption_t1(probas, config):
    """
    Calculates probabilities of the different levels of market adoption.

    Params:
        - config {dict}: dictionary containing the changable parameters of the model

    Returns:
        - market_adoption_probas {dict}: dictionary with probabilities
                for marketadoption to be low, neutral or high

    """

    # setup
    weighting = config["MA_inputweights"]

    # get conditional probabilities
    cat = {0: "negative", 1: "neutral", 2: "positive"}
    df = market_adaption_cond_probas(weighting)

    # calculate joint probability for each combination
    public_perception_p = df.public_perception.apply(
        lambda x: probas["public_perception"][cat[x]]
    )
    reg_p = df.reg.apply(lambda x: probas["reg"][cat[x]])
    technology_p = df.technology.apply(lambda x: probas["technology"][cat[x]])
    df["joint_prior"] = public_perception_p * reg_p * technology_p

    df["joint_ma_low"] = df.apply(
        lambda row: row.joint_prior * row.market_adoption_cond_proba[0], axis=1
    )
    df["joint_ma_neutral"] = df.apply(
        lambda row: row.joint_prior * row.market_adoption_cond_proba[1], axis=1
    )
    df["joint_ma_high"] = df.apply(
        lambda row: row.joint_prior * row.market_adoption_cond_proba[2], axis=1
    )

    # marginalize over all combinations
    market_adoption_probas = {
        "low": round(df.joint_ma_low.sum(), 4),
        "neutral": round(df.joint_ma_neutral.sum(), 4),
        "high": round(df.joint_ma_high.sum(), 4),
    }

    return market_adoption_probas


def market_adaption_t2(ma_t1_probas):
    """Returns probabilities for market adoption levels in year 2."""
    ma_t2_probas = deepcopy(ma_t1_probas)
    return ma_t2_probas


def update_base_proba(coin_base_probas, ma_probas):
    """
    Calculates probabilities for levels of returns for a given coin, using
    the base return of the coin and the general market adaption.

    Params:
        - coin_base_probas {dict}: base probability for return levels of a particular coin
                                    can be aquired using the coin_base() function.
        - ma_probas {dict}: probabilities for possible values of market adaption.
                            can be aquired with market_adaption_t1() or market_adaption_t2()
    """
    ma_high = [x * ma_probas["high"] for x in coin_base_probas.values()]
    ma_neutral = [x * ma_probas["neutral"] for x in coin_base_probas.values()]
    ma_low = [x * ma_probas["low"] for x in coin_base_probas.values()]

    p = dict()
    p["low_low"] = ma_neutral[0] + ma_low[0] + ma_low[1]
    p["low"] = ma_neutral[1] + ma_low[2] + ma_high[0]
    p["neutral"] = ma_neutral[2] + ma_low[3] + ma_high[1]
    p["high"] = ma_neutral[3] + ma_low[4] + ma_high[2]
    p["high_high"] = ma_neutral[4] + ma_high[3] + ma_high[4]

    return p


def calculate_probabilities(config, custom_p=dict()):
    p = dict()

    p["stock_t1"] = custom_p.get("stock_t1", get_probas("STOCKMARKET_T1", config))
    p["stock_t2"] = custom_p.get("stock_t2", get_probas("STOCKMARKET_T2", config))
    p["reg"] = custom_p.get("reg", get_probas("REG", config))
    p["public_perception"] = custom_p.get(
        "public_perception", get_probas("PUBLIC_PERCEPTION", config)
    )
    p["technology"] = custom_p.get("technology", get_probas("TECHNOLOGY", config))

    p["ma_t1"] = custom_p.get("ma_t1", market_adaption_t1(p, config))
    p["ma_t2"] = custom_p.get("ma_t2", market_adaption_t2(p["ma_t1"]))

    for coin in ["BTC", "ETH", "SOL"]:
        base_t1_probas = coin_base(f"{coin}", p["stock_t1"], config)
        p[f"{coin}_t1"] = custom_p.get(
            f"{coin}_t1", update_base_proba(base_t1_probas, p["ma_t1"])
        )

        base_t2_probas = coin_base(f"{coin}", p["stock_t2"], config)
        p[f"{coin}_t2"] = custom_p.get(
            f"{coin}_t2", update_base_proba(base_t2_probas, p["ma_t2"])
        )

    return p
