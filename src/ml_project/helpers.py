from functools import cached_property

import numpy as np
import pandas as pd

VAL_TYPES = [-1, 0, 1]


class ValTypes:
    '''
    Analyze value types and logical relationships between features in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        ignore (list[str], optional): List of columns to ignore in the analysis.
    '''
    def __init__(self, df: pd.DataFrame, ignore: list[str] = []):
        '''
        Initialize the ValTypes object.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.
            ignore (list[str], optional): List of columns to ignore.
        '''
        self.df = df
        self.ignore = ignore

    @cached_property
    def types(self):
        '''
        Compute all unique value type patterns in the DataFrame (ignoring specified columns).

        Returns:
            pd.DataFrame: DataFrame where each row is a unique pattern of value types (-1, 0, 1) and a count.
        '''
        return (
            self.df.drop(columns=self.ignore)
            .map(lambda x: -1 if x < 0 else 0 if x == 0 else 1)
            .value_counts()
            .reset_index()
        )
    
    def df_of_type(self, idx):
        '''
        Return the subset of the DataFrame matching the value type pattern at the given index.

        Args:
            idx (int): Index of the value type pattern in self.types.

        Returns:
            pd.DataFrame: Subset of the DataFrame matching the pattern.
        '''
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
    def types_counts(self):
        '''
        Count the occurrences of each value type (-1, 0, 1) for each feature.

        Returns:
            pd.DataFrame: Grouped counts of value types per feature.
        '''
        return (
            self.types.melt(
                id_vars=["count"], value_vars=list(set(self.types.columns) - {"count"})
            )
            .groupby(["variable", "value"])
            .sum()
        )

    @cached_property
    def unique_types(self):
        '''
        List the unique value types present for each feature.

        Returns:
            pd.Series: Series mapping feature names to arrays of unique value types.
        '''
        return (
            self.types.drop(columns="count")
            .transpose()
            .apply(lambda x: x.unique(), axis=1)
        )

    @cached_property
    def impls(self):
        '''
        Compute logical implication relationships between feature value types.

        Returns:
            dict: Nested dictionary of the form {if_feat: {if_val: {then_val: [then_feat, ...]}}}
                  representing which features are implied by others for each value type.
        '''
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
        '''
        Convert the logical implication relationships into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame with columns [if_feat, if_val, then_feat, then_val].
        '''
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
        '''
        Find equivalence groups of features based on logical implication relationships.

        Returns:
            list: List of sets, each set contains tuples (feature, value_type) that are equivalent.
        '''
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
