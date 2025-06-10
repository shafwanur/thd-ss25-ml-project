from functools import cached_property

import numpy as np
import pandas as pd

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
