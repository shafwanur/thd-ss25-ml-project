import numpy as np
import pandas as pd

VAL_TYPES = [-1, 0, 1]


def to_val_types(df):
    return (
        df.map(lambda x: -1 if x < 0 else 0 if x == 0 else 1)
        .value_counts()
        .reset_index()
    )


def get_connections(val_types):
    """returns dictionary: if_feat => if_val => then_val => then_feat"""

    df = val_types
    features = df.drop(columns="count")

    connections = {}

    for feature in features:
        imp = {}

        for val in VAL_TYPES:
            for implied_val in VAL_TYPES:
                d = df.loc[df[feature] == val].apply(lambda x: x == implied_val)
                if d.shape[0] == 0:
                    continue
                implied = d.all()
                implied[feature] = False
                imp_val = imp.setdefault(val, {})
                implied_list = list(implied[implied].index)
                if not implied_list:
                    continue
                imp_val[implied_val] = implied_list
        connections[feature] = imp
    return connections


def get_connections_df(connections):
    """turns the connections dictionary into dataframe"""

    tuples = []

    for f, i1 in connections.items():
        for v, i2 in i1.items():
            for vc, fcs in i2.items():
                for fc in fcs:
                    tuples.append((f, v, fc, vc))

    return pd.DataFrame(tuples, columns=["if_feat", "if_val", "then_feat", "then_val"])


def get_connected(connections, feature, val, implied_val):
    return connections.get(feature, {}).get(val, {}).get(implied_val, [])


def get_connection_groups(connections):
    """returns a list of equivalence classes based on the connections dictionary"""

    eq_groups = []

    to_check = {(feature, value) for feature in connections for value in VAL_TYPES}

    while to_check:
        fv = to_check.pop()
        feature, val = fv
        eqs = {fv}

        for implied_val in VAL_TYPES:
            for connected_feature in get_connected(
                connections, feature, val, implied_val
            ):
                if feature in get_connected(
                    connections, connected_feature, implied_val, val
                ):
                    to_check.discard((connected_feature, implied_val))
                    eqs.add((connected_feature, implied_val))
        if len(eqs) <= 1:
            continue
        eq_groups.append(eqs)

    return sorted(eq_groups)


def describe(df: pd.DataFrame):
    desc = df.describe()
    desc_t = desc.transpose()

    return (
        desc.apply(np.log)
        .map(lambda x: -np.inf if x == -np.inf else x * 10 // 1 / 10)
        .assign(clas=desc["clas"])
        .transpose()
        .assign(count=desc_t["count"])
    )
