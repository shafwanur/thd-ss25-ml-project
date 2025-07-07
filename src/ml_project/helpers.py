import math
from functools import cached_property

import numpy as np
import pandas as pd
from IPython.display import display

VAL_TYPES = [-1, 0, 1]


class ValTypes:
    def __init__(self, df: pd.DataFrame, ignore: list[str] = []):
        self.df = df
        self.ignore = ignore

    def df_of_type(self, idx):
        typ = self.types.drop(columns="count").iloc[idx]
        cond = True
        for feat in typ.index:
            val = typ[feat]
            if val < 0:
                cond = (self.df[feat] < 0) & cond
            elif val == 0:
                cond = (self.df[feat] == 0) & cond
            else:
                cond = (self.df[feat] > 0) & cond
        return self.df.loc[cond]

    @cached_property
    def types(self):
        return (
            self.df.drop(columns=self.ignore)
            .map(lambda x: -1 if x < 0 else 0 if x == 0 else 1)
            .value_counts()
            .reset_index()
        )

    @cached_property
    def types_counts(self):
        return (
            self.types.melt(
                id_vars=["count"], value_vars=list(set(self.types.columns) - {"count"})
            )
            .groupby(["variable", "value"])
            .sum()
        )

    @cached_property
    def unique_types(self):
        return (
            self.types.drop(columns="count")
            .transpose()
            .apply(lambda x: x.unique(), axis=1)
        )

    @cached_property
    def impls(self):
        """returns dictionary: if_feat => if_val => then_val => then_feat"""

        df = self.types
        feats = df.drop(columns="count")

        impls = {}

        for feat in feats:
            imp = {}

            for val in VAL_TYPES:
                for implied_val in VAL_TYPES:
                    d = df.loc[df[feat] == val].apply(lambda x: x == implied_val)
                    if d.shape[0] == 0:
                        continue
                    implied = d.all()
                    implied[feat] = False
                    imp_val = imp.setdefault(val, {})
                    implied_list = list(implied[implied].index)
                    if not implied_list:
                        continue
                    imp_val[implied_val] = implied_list
            impls[feat] = imp
        return impls

    @cached_property
    def impls_df(self):
        """turns the connections dictionary into dataframe"""

        tuples = []

        for f, i1 in self.impls.items():
            for v, i2 in i1.items():
                for vc, fcs in i2.items():
                    for fc in fcs:
                        tuples.append((f, v, fc, vc))

        return pd.DataFrame(
            tuples, columns=pd.Index(["if_feat", "if_val", "then_feat", "then_val"])
        )

    @cached_property
    def eq_groups(self):
        """returns a list of equivalence classes based on the connections dictionary"""

        def get_connected(feature, val, implied_val):
            return self.impls.get(feature, {}).get(val, {}).get(implied_val, [])

        eq_groups = []

        to_check = {(feature, value) for feature in self.impls for value in VAL_TYPES}

        while to_check:
            fv = to_check.pop()
            feat, val = fv
            eqs = {fv}

            for implied_val in VAL_TYPES:
                for connected_feat in get_connected(feat, val, implied_val):
                    if feat in get_connected(connected_feat, implied_val, val):
                        to_check.discard((connected_feat, implied_val))
                        eqs.add((connected_feat, implied_val))
            if len(eqs) <= 1:
                continue
            eq_groups.append(eqs)

        return sorted(eq_groups)


def stats(value: str):
    return [f"min_{value}", f"max_{value}", f"mean_{value}", f"std_{value}"]


def fix_mean_rounding(df: pd.DataFrame, value: str):
    """For some rows, when min == max, mean is rounded to 0.1 seconds.
    Fixes this.
    """

    print(f"fix mean rounding for {value}")

    min, max, mean, std = stats(value)
    cond = (df[min] == df[max]) & (df[min] != df[mean])

    # Ensure the case
    loc = df.loc[cond, [min, max, mean, std]]
    assert check_condition(loc, loc[std].eq(0), f"{std} == 0")
    rounded_cond = loc[min].map(round_time) == loc[mean]
    assert check_condition(loc, rounded_cond, f"{mean} == rount_time({min})")
    check_condition(loc, cond, "min == max and min != mean")

    # Return updated df
    res = df.copy(deep=False)
    res.loc[cond, mean] = res[min]
    return res


PERCS = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


def s(x: float):
    return math.ceil(x * 1000000) // 1


SEC = s(1)


def to_sec(x: float):
    if x < 0:
        return x
    else:
        return x / SEC


def desc_time(df: pd.DataFrame):
    return df.map(to_sec).describe(PERCS).transpose()


def hist_time(df: pd.DataFrame, bins: int = 15):
    return df.map(to_sec).hist(bins=bins)


def is_rounded(x: float):
    """Some values in the data are rounded to 0.1s"""

    return x == round_time(x)


def round_time(x: float):
    if x < 0:
        return x
    else:
        return round(x, -5)


def assert_stat_invariants(df: pd.DataFrame, value: str, with_std: bool = True) -> None:
    min, max, mean, std = stats(value)
    ok = True

    ok = check_condition(df, df[min] <= df[mean], f"{min} <= {mean}") and ok
    ok = check_condition(df, df[mean] <= df[max], f"{mean} <= {max}") and ok
    if std:
        ok = (
            check_condition(
                df,
                ((df[min] != df[max]) | (df[std] == 0)),
                f"{min} == {max} => {std} == 0",
            )
            and ok
        )
        ok = (
            check_condition(
                df,
                ((df[min] == df[max]) | (df[std] != 0)),
                f"{std} == 0 => {min} == {max}",
            )
            and ok
        )

    if not ok:
        raise AssertionError


def check_condition(df: pd.DataFrame, cond: pd.Series, msg: str) -> bool:
    """Check that the condition holds, otherwise - display the wrong rows"""

    if cond.all():
        print(f"OK: {msg}")
        return True

    print(f"FAIL: {msg}")
    display(df.loc[~cond])
    return False
