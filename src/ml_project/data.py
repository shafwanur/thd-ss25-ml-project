from functools import cache, cached_property
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff

pd.set_option("future.no_silent_downcasting", True)
pd.set_option("mode.copy_on_write", True)


class Dataset:
    def __init__(self, path: Path):
        self.path = path

    @cached_property
    def _arff(self):
        return loadarff(self.path)

    @cached_property
    def cls_list(self):
        return list(map(str.encode, self._arff[1]["class1"][1]))

    @cached_property
    def cls_id_map(self):
        return {c: index + 1 for index, c in enumerate(self.cls_list)}

    @cached_property
    def cls_id_repr_map(self):
        return {
            index + 1: f"{c.decode()} ({index + 1})"
            for index, c in enumerate(self.cls_list)
        }

    @cached_property
    def raw(self):
        df = pd.DataFrame(self._arff[0])
        return df

    @cached_property
    def orig(self):
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
    def with_dur(self) -> pd.DataFrame:
        # The rows with duration == 0 seem to be trash
        df = self.orig
        return df.loc[df.duration > 0]

    @property
    def mostly_present(self):
        # These 5 features are always here (when duration > 0)
        return [
            "duration",
            "pps",
            "bps",
            "max_flowiat",
            "mean_flowiat",
        ]

    @cached_property
    def numerical5(self):
        return self.with_dur[self.mostly_present + ["cls"]]

    @cached_property
    def drop_missing(self):
        return positive(self.orig)

    @cached_property
    def flagged(self):
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
    desc = df.describe()
    desc_t = desc.transpose()

    return (
        desc.apply(np.log)
        .map(lambda x: -np.inf if x == -np.inf else x * 10 // 1 / 10)
        .assign(cls=desc["cls"])
        .transpose()
        .assign(count=desc_t["count"])
    )


def positive(df: pd.DataFrame):
    cond = (df > 0).apply("any", axis=1)
    return df.loc[cond]
