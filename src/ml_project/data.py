from pathlib import Path
from functools import cache, cached_property

import numpy as np
import pandas as pd
import dataframe_image as dfi
from scipy.io.arff import loadarff

pd.set_option("future.no_silent_downcasting", True)
pd.set_option("mode.copy_on_write", True)


class Dataset:
    def __init__(self, path: Path):
        '''
        Initialize the Dataset object.

        Args:
            path (Path): Path to the ARFF file to load.
        '''
        self.path = path

    @cached_property
    def _arff(self):
        '''
        Load the ARFF file from the specified path.

        Returns:
            tuple: Data and metadata loaded from the ARFF file.
        '''
        return loadarff(self.path)

    @cached_property
    def cls_list(self): # TODO: why are the labels being saved as bytes? 
        '''
        Get the list of class labels as bytes.

        Returns:
            list: List of class labels in bytes format.
        '''
        return list(map(str.encode, self._arff[1]["class1"][1]))

    @cached_property
    def cls_id_map(self):
        '''
        Map class labels to integer IDs starting from 1.

        Returns:
            dict: Mapping from class label to integer ID.
        '''
        return {c: index + 1 for index, c in enumerate(self.cls_list)}

    @cached_property
    def cls_id_repr_map(self):
        '''
        Map integer class IDs to human-readable string representations.

        Returns:
            dict: Mapping from integer ID to string label.
        '''
        return {
            index + 1: f"{c.decode()} ({index + 1})"
            for index, c in enumerate(self.cls_list)
        }

    @cached_property
    def raw(self):
        '''
        Return the raw DataFrame loaded from the ARFF file.

        Returns:
            pd.DataFrame: Raw data as a DataFrame.
        '''
        df = pd.DataFrame(self._arff[0])
        return df

    @cached_property
    def orig(self):
        '''
        Return the original DataFrame with three columns (class1, flowPktsPerSecond, and flowBytesPerSecond) renamed and integer class labels.

        Returns:
            pd.DataFrame: DataFrame with renamed columns and integer class labels instead of strings.
        '''
        df = self.raw
        df = df.rename(
            columns={
                "class1": "cls",
                "flowPktsPerSecond": "pps",
                "flowBytesPerSecond": "bps",
            }
        )
        df["cls"] = df["cls"].replace(self.cls_id_map).astype(int)
        return df

    @cached_property
    def with_dur(self):
        '''
        Return the DataFrame filtered to only rows with duration > 0.

        Returns:
            pd.DataFrame: Filtered DataFrame with duration > 0.
        '''
        df = self.orig
        return df.loc[df.duration > 0]

    @property
    def mostly_present(self):
        '''
        List of the five features that are always present when duration > 0.

        Returns:
            list: List of feature names.
        '''
        return [
            "duration",
            "pps",
            "bps",
            "max_flowiat",
            "mean_flowiat",
        ]

    @cached_property
    def numerical5(self):
        '''
        Return a DataFrame with the five mostly present numerical features and class label.

        Returns:
            pd.DataFrame: DataFrame with selected features and class label.
        '''
        return self.with_dur[self.mostly_present + ["cls"]]

    @cached_property
    def drop_missing(self):
        '''
        Return a DataFrame with only rows where *at least* one value is positive.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        '''
        return positive(self.orig)

    @cached_property
    def flagged(self):
        '''
        Return a DataFrame with selected features and binary flags for feature presence for classes where the majority are missing.

        Returns:
            pd.DataFrame: DataFrame with selected features and binary indicator columns.
        '''
        df = self.with_dur

        return pd.DataFrame(
            {
                "cls": df.cls,
                "duration": df.duration,
                "pps": df.pps,
                "bps": df.bps,
                "max_flowiat": df.max_flowiat,
                "mean_flowiat": df.mean_flowiat,
                # The *_active *_idle features are missing in more than half of the rows
                # And their missing or being present are connected
                "has_active": (df.max_active > 0).astype(int),
                # The std_active and std_idle behave a bit differently from *_active
                "has_std_active": (df.std_active > 0).astype(int),
                # TODO: the following features require more inspection
                "has_fiat": (df.max_fiat > 0).astype(int),
                "has_biat": (df.max_biat > 0).astype(int),
                "has_min_flowiat": (df["min_flowiat"] >= 0).astype(int),
                "has_std_flowiat": (df["std_flowiat"] > 0).astype(int),
                "has_mean_fiat": (df["mean_fiat"] > 0).astype(int),
                "has_mean_biat": (df["mean_biat"] > 0).astype(int),
            }
        )

    def class_fraction(
        self, df_num: pd.DataFrame, df_den: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        '''
        Calculate class-wise population statistics for a subset of the data.

        Args:
            df_num (pd.DataFrame): The numerator DataFrame (subset to analyze), must contain a 'cls' column.
            df_den (pd.DataFrame, optional): The denominator DataFrame (reference set for relative fractions).
                If None, uses the full dataset (self.orig).

        Returns:
            pd.DataFrame: A DataFrame with the following columns for each class:
                - pop: Number of samples in df_num for each class.
                - pop_f: Percentage of each class in df_num relative to the total in df_num.
                - f: Percentage of each class in df_num relative to its count in df_den.
            Includes an additional row ("all") with totals for all classes.
        '''
        if df_den is None:
            df_den = self.orig

        population = df_num.groupby("cls").size()
        population_den = df_den.groupby("cls").size()
        whole_population = population.sum()
        whole_population_den = population_den.sum()
        population_f = (population / whole_population * 100).round(1)
        fraction = (population / population_den * 100).round(1)

        return pd.concat(
            [
                pd.DataFrame(
                    {
                        "pop": population,
                        "pop_f": population_f,
                        "f": fraction,
                    }
                ).rename(self.cls_id_repr_map),
                pd.DataFrame(
                    data=[
                        (
                            whole_population,
                            100,
                            round(whole_population / whole_population_den * 100, 1),
                        )
                    ],
                    columns=pd.Index(["pop", "pop_f", "f"]),
                    index=pd.Index(["all"]),
                ),
            ]
        )


def describe(df: pd.DataFrame):
    '''
    Generate a custom description of the DataFrame, including log-transformed statistics.

    Args:
        df (pd.DataFrame): The DataFrame to describe.

    Returns:
        pd.DataFrame: Transposed DataFrame with log-transformed statistics and counts.
    '''
    desc = df.describe()
    desc_t = desc.transpose()

    return (
        desc.apply(np.log)
        .map(lambda x: -np.inf if x == -np.inf else x * 10 // 1 / 10)
        .assign(clas=desc["cls"])
        .transpose()
        .assign(count=desc_t["count"])
    )


def positive(df: pd.DataFrame):
    '''
    Filter the DataFrame to only rows where at least one value is positive.

    Args:
        df (pd.DataFrame): The DataFrame to filter.

    Returns:
        pd.DataFrame: Filtered DataFrame with at least one positive value per row.
    '''
    cond = (df > 0).apply("any", axis=1)
    return df.loc[cond]

def save(df: pd.DataFrame, file_name: str):
    '''
    Save a dataframe as an image in the img folder

    Args: 
        df (pd.DataFrame): The dataframe to save 
        file_name (str) : the name of the file. 
    '''
    dfi.export(df, f"../img/{file_name}", table_conversion="chrome")