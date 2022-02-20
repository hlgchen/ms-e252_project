import pandas as pd
import yaml
import itertools
from probability import *
from copy import deepcopy
from pprint import pprint


def get_return(decision, p, config, verbose=True):
    """
    Returns the expected value of a deal given decisions made.

    Params:
        - decision {tuple/list}: should contain two strings. The string in position 0 is
                                the option chosen at t=0, the string at position 1 is the
                                option chosen at time t=1.
        - p {dict}: dictionary containing probabilities for possible values of
                    all uncertainties
        - config {dict}: config dictionary, which also contains the mapping from
                        return levels to actual return numbers
        - verbose {bool}: If true the probability for each return outcome is printed out

    Returns:
        - ev {float}: expected value of deal given decision

    """
    choice1 = decision[0]
    choice2 = decision[1]

    p1 = p2 = {"inflation": 1}
    if choice1 != "CASH":
        p1 = p[f"{choice1}_t1"]
    if choice2 != "CASH":
        p2 = p[f"{choice2}_t2"]

    probas = dict()
    for return_level1, p_return_level1 in p1.items():
        for return_level2, p_return_level2 in p2.items():
            rl = return_level1 + "|" + return_level2
            probas[(rl, config[return_level1] * config[return_level2])] = (
                p_return_level1 * p_return_level2
            )
    if verbose:
        print(probas)
    ev = sum([k[1] * v for k, v in probas.items()])
    return ev


def get_all_returns(probabilities, config):
    """
    Returns dataframe with EV for all possible decision combinations.

    Params:
        - probabilities {dict}: dictionary containing probabilities for possible values of
                    all uncertainties
        - config {dict}: config dictionary, which also contains the mapping from
                        return levels to actual return numbers

    Returns:
        - df {pd.DataFrame}: first column is choice (that is the decision combination).
                            second column is ev (that is the expected value for the decision)
                            sorted in descending order by EV.

    """
    options = list(
        itertools.product(["BTC", "ETH", "SOL", "CASH"], ["BTC", "ETH", "SOL", "CASH"])
    )
    df = pd.DataFrame()
    for i, decision in enumerate(options):
        df.loc[i, "choice"] = str(decision)
        df.loc[i, "ev"] = get_return(decision, probabilities, config, verbose=False)
    df = df.sort_values(by="ev", ascending=False)
    return df


def get_deal_value(probabilities, config, verbose=True):
    """
    Returns the best action and the corresponding EV.

    Params:
        - probabilities {dict}: dictionary containing probabilities for possible values of
                    all uncertainties
        - config {dict}: config dictionary, which also contains the mapping from
                        return levels to actual return numbers
        - verbose {bool}: if True prints out the best action and EV

    Returns:
        - df.iloc[0, 0] {tuple[str, str]}: best action tuple
        - df.iloc[0, 1] {float}: corresponding EV
    """
    df = get_all_returns(probabilities, config)
    if verbose:
        print(f"best action is: {df.iloc[0, 0]} with EV of {df.iloc[0, 1]}")
    return df.iloc[0, 0], df.iloc[0, 1]


def clairvoyance(X, probabilities, config):
    """
    Returns value of deal when having clairvoyance on X and optimal decision and EV given a X takes on a certain value.

    Params:
        - X {str}: string specifying the
        - probabilities {dict}: dictionary containing probabilities for possible values of
                    all uncertainties
        - config {dict}: config dictionary, which also contains the mapping from
                        return levels to actual return numbers

    Returns:
         - deal_value_with_cv {float}:
         - action_values {dict}:

    """
    p_X_base = deepcopy(probabilities[X])
    for outcome in p_X_base.keys():
        # set all probabilities for X outcomes to 0
        p_X_base[outcome] = 0

    best_action_values = dict()
    for outcome in p_X_base.keys():
        p_X = {X: deepcopy(p_X_base)}
        p_X[X][outcome] = 1
        hypothetical_probabilities = calculate_probabilities(config, p_X)
        action, value = get_deal_value(
            hypothetical_probabilities, config, verbose=False
        )
        best_action_values[outcome] = (action, value)

    deal_value_with_cv = 0
    for outcome in p_X_base.keys():
        deal_value_with_cv += probabilities[X][outcome] * best_action_values[outcome][1]

    result = dict()
    result["deal_value_free_cv"] = deal_value_with_cv
    result["cv_value_with_delta"] = (
        deal_value_with_cv - get_deal_value(probabilities, config)[1]
    )
    result["best_action_values"] = best_action_values

    return result


if __name__ == "__main__":

    with open("dtree_config.yml", "r") as stream:
        config = yaml.safe_load(stream)

    probabilities = calculate_probabilities(config)

    pprint(clairvoyance("BTC_t1", probabilities, config))

    # print(get_all_returns(probabilities, config))
