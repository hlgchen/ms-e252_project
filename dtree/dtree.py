import pandas as pd
import numpy as np
import yaml
import itertools
from probability import *
from copy import deepcopy
from pprint import pprint
from pathlib import Path
import os


def get_project_root():
    """Returns absolute path of project root."""
    return Path(__file__).parent.parent


def get_return(decision, p, config, ux, verbose=True):
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
        - ce {float}: certain equivalent of deal given decision
        - probas {dict}: probabilities of different scenarios and the associated returns

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
            total_return = config[return_level1] * config[return_level2]
            absolute_return = (total_return - 1) * config["investment_amount"]
            u_value = ux[0](absolute_return)
            probas[(rl, u_value, absolute_return, total_return)] = (
                p_return_level1 * p_return_level2
            )
    probas = dict(sorted(probas.items(), key=lambda item: item[0][1], reverse=True))
    if verbose:
        print(probas)

    ev = sum([k[1] * v for k, v in probas.items()])
    ce = ux[1](ev)
    return ce, probas


def get_all_returns(probabilities, config, ux, detailed=False):
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
                            The subsequent columns are different scenarios with each entry being
                            a string describing the scenario, the associated return and the
                            probability of that scenario occuring.

    """
    options = list(
        itertools.product(["BTC", "ETH", "SOL", "CASH"], ["BTC", "ETH", "SOL", "CASH"])
    )
    df = pd.DataFrame()
    for i, decision in enumerate(options):
        df.loc[i, "choice"] = str(decision)
        df.loc[i, "ce"], probas = get_return(
            decision, probabilities, config, ux, verbose=False
        )
        if detailed:
            for j, (k, v) in enumerate(probas.items()):
                s = f"scenario ({k[0]}) with u-value {round(k[1], 4)}"
                s += f" (return abs: {round(k[2], 4)}, rel:{round(k[3], 4)}) has probability {v}"
                df.loc[i, f"scenario{j}"] = s
    df = df.sort_values(by="ce", ascending=False).reset_index(drop=True)
    return df


def get_deal_value(probabilities, config, ux, verbose=True):
    """
    Returns the best action and the corresponding CE.

    Params:
        - probabilities {dict}: dictionary containing probabilities for possible values of
                    all uncertainties
        - config {dict}: config dictionary, which also contains the mapping from
                        return levels to actual return numbers
        - verbose {bool}: if True prints out the best action and CE

    Returns:
        - df.iloc[0, 0] {tuple[str, str]}: best action tuple
        - df.iloc[0, 1] {float}: corresponding CE
    """
    df = get_all_returns(probabilities, config, ux)
    if verbose:
        print(f"best action is: {df.loc[0, 'choice']} with CE of {df.loc[0, 'ce']}")
    return df.iloc[0, 0], df.iloc[0, 1]


def clairvoyance(X, probabilities, config, ux):
    """
    Returns value of deal when having clairvoyance on X and optimal decision and CE given a X takes on a certain value.

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
            hypothetical_probabilities, config, ux, verbose=False
        )
        best_action_values[outcome] = (action, value)

    deal_value_with_cv = 0
    for outcome in p_X_base.keys():
        deal_value_with_cv += probabilities[X][outcome] * best_action_values[outcome][1]

    result = dict()
    result["deal_value_free_cv"] = deal_value_with_cv
    result["cv_value_with_delta"] = (
        deal_value_with_cv - get_deal_value(probabilities, config, ux, verbose=False)[1]
    )
    result["best_action_values"] = best_action_values

    return result


def get_path(path):
    return os.path.join(get_project_root(), path)


def get_ux(risk_tolerance):
    def u(x):
        return -np.exp(-x / risk_tolerance)

    def x(u):
        return -risk_tolerance * np.log(-u)

    return u, x


if __name__ == "__main__":
    with open(get_path("dtree/dtree_config.yml"), "r") as stream:
        config = yaml.safe_load(stream)

    ux = get_ux(4000)
    probabilities = calculate_probabilities(config)
    decision_ces = get_all_returns(probabilities, config, ux)
    decision_details = get_all_returns(probabilities, config, ux, detailed=True)

    uncertainties = [
        "stock_t1",
        "stock_t2",
        "reg",
        "public_perception",
        "technology",
        "ma_t1",
        "ma_t2",
        "BTC_t1",
        "ETH_t1",
        "SOL_t1",
        "BTC_t2",
        "ETH_t2",
        "SOL_t2",
    ]
    cv = dict()
    for uncertainty in uncertainties:
        cv[uncertainty] = clairvoyance(uncertainty, probabilities, config, ux)
    cv_concise = {k: round(v["cv_value_with_delta"], 4) for k, v in cv.items()}

    with open(get_path("dtree/outputs/probabilities.txt"), "w") as f:
        pprint(probabilities, stream=f)

    with open(get_path("dtree/outputs/decision_ces.txt"), "w") as f:
        pprint(decision_ces, stream=f)

    with open(get_path("dtree/outputs/decision_details.txt"), "w") as f:
        s = ""
        ncol = decision_details.shape[1]
        nrow = decision_details.shape[0]
        for i in range(nrow):
            s += f"{decision_details.loc[i, 'choice']} has ce : {decision_details.loc[i, 'ce']} \n"
            for j in range(2, ncol):
                detail = decision_details.iloc[i, j]
                if isinstance(detail, str):
                    s += "\t" + detail + "\n"
            s += "\n\n"

        f.write(s)

    with open(get_path("dtree/outputs/clairvoyance.txt"), "w") as f:
        pprint(cv, stream=f, width=120)

    with open(get_path("dtree/outputs/clairvoyance_consice.txt"), "w") as f:
        pprint(cv_concise, stream=f, width=120)
