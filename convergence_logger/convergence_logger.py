"""
Module for tracking convergence statistics of a series of values over time.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes


################################
## Types of Statistics supported
################################
class IntervalStatistics(ABC):
    """
    Static class to compute several common statistics over intervals.

    Used by `Series_Statistics`, enabling this class to be independent of the statistics used.
    """

    @classmethod
    @abstractmethod
    def reset(cls, storage: np.ndarray) -> None:
        """
        Resets the storage array for statistics.

        Args:
            storage (np.ndarray): The array to reset.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def __len__(cls) -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def add_value(storage: np.ndarray, value: float) -> None:
        """
        Adds a value to the statistics storage.

        Args:
            storage (np.ndarray): The statistics storage array.
            value (float): The value to add.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def merge_intervals(
        storage1: np.ndarray, storage2: np.ndarray, new_storage: np.ndarray
    ) -> None:
        """
        Merges statistics from two intervals into a new storage.

        Args:
            storage1 (np.ndarray): Statistics from the first interval.
            storage2 (np.ndarray): Statistics from the second interval.
            new_storage (np.ndarray): The storage for the merged statistics.
        """
        raise NotImplementedError


COUNT: int = 0
MIN: int = 1
MAX: int = 2
MEAN: int = 3
MEAN_SQR: int = 4


class CountMinMaxMeanVarStd(IntervalStatistics):
    """
    IntervalStatistics class that computes: count, min,max, mean, mean_sqr, var, std.
    """

    names: dict[str, int] = {
        "count": COUNT,
        "min": MIN,
        "max": MAX,
        "mean": MEAN,
        "mean_sqr": MEAN_SQR,
    }
    n_statistics: int = len(names)

    @classmethod
    def reset(cls, storage: np.ndarray) -> None:
        if storage.shape != (len(cls.names),):
            raise ValueError("storage does not match shape for statistics")
        storage[COUNT] = 0.0
        storage[MIN] = np.inf
        storage[MAX] = -np.inf
        storage[MEAN] = np.nan
        storage[MEAN_SQR] = np.nan

    @classmethod
    def __len__(cls) -> int:
        return len(cls.names)

    @staticmethod
    def add_value(storage: np.ndarray, value: float) -> None:
        if np.isnan(value):
            # nan would mess up accumulated intervals
            return
        old_count = storage[COUNT]
        new_count = old_count + 1
        ratio = old_count / new_count
        storage[COUNT] = new_count
        if new_count <= 1:
            storage[MIN] = value
            storage[MAX] = value
            storage[MEAN] = value
            storage[MEAN_SQR] = value**2
        else:
            storage[MIN] = min(storage[MIN], value)
            storage[MAX] = max(storage[MAX], value)
            storage[MEAN] = ratio * storage[MEAN] + value / new_count
            storage[MEAN_SQR] = ratio * storage[MEAN_SQR] + value**2 / new_count

    @staticmethod
    def merge_intervals(
        storage1: np.ndarray, storage2: np.ndarray, new_storage: np.ndarray
    ) -> None:
        count1 = storage1[COUNT]
        count2 = storage2[COUNT]
        new_count = count1 + count2
        new_storage[COUNT] = new_count
        new_storage[MIN] = min(storage1[MIN], storage2[MIN])
        new_storage[MAX] = max(storage1[MAX], storage2[MAX])
        if new_count > 0:
            if count1 > 0 and count2 > 0:
                ratio1 = count1 / new_count
                ratio2 = count2 / new_count
                new_storage[MEAN] = ratio1 * storage1[MEAN] + ratio2 * storage2[MEAN]
                new_storage[MEAN_SQR] = (
                    ratio1 * storage1[MEAN_SQR] + ratio2 * storage2[MEAN_SQR]
                )
            elif count1 > 0:
                new_storage[MEAN] = storage1[MEAN]
                new_storage[MEAN_SQR] = storage1[MEAN_SQR]
            else:
                new_storage[MEAN] = storage2[MEAN]
                new_storage[MEAN_SQR] = storage2[MEAN_SQR]
        else:
            new_storage[MEAN] = np.nan
            new_storage[MEAN_SQR] = np.nan

    @classmethod
    def plot_mean(
        cls,
        axis: matplotlib.axes.Axes,
        t: np.ndarray,
        values: np.ndarray,
        *,
        label: str = "mean",
        color: str | None = None,
    ) -> None:
        """
        Plots the means of the values.

        Args:
            axis (matplotlib.axes.Axes): The axis to plot on.
            t (np.ndarray): The time points.
            values (np.ndarray): The statistics values.
            label (str, optional): The label for the plot. Defaults to "mean".
            color (str | None, optional): The color of the plot. Defaults to None.
        """
        y = values[:, MEAN]
        good = ~np.isnan(y)
        if not good.any():
            return
        t_good = t[good]
        y_good = y[good]
        if len(t_good) <= 75:
            s = max(75 / len(t_good), 1) ** 2
            if color is None:
                axis.scatter(t_good, y_good, s=s, label=label)
            else:
                axis.scatter(t_good, y_good, s=s, label=label, color=color)
        else:
            if color is None:
                axis.plot(t_good, y_good, label=label)
            else:
                axis.plot(t_good, y_good, label=label, color=color)

    @classmethod
    def plot_range(
        cls,
        axis: matplotlib.axes.Axes,
        t: np.ndarray,
        values: np.ndarray,
        *,
        label: str = "min/max",
        color: str | None = None,
        alpha: float = 0.3,
    ) -> None:
        """
        Plots the ranges (min/max) of the values.

        Args:
            axis (matplotlib.axes.Axes): The axis to plot on.
            t (np.ndarray): The time points.
            values (np.ndarray): The statistics values.
            label (str, optional): The label for the plot. Defaults to "min/max".
            color (str | None, optional): The color of the plot. Defaults to None.
            alpha (float, optional): The transparency of the fill. Defaults to 0.3.
        """
        lower = values[:, MIN]
        upper = values[:, MAX]
        if color is None:
            axis.fill_between(t, lower, upper, alpha=alpha, label=label)
        else:
            axis.fill_between(t, lower, upper, alpha=alpha, label=label, color=color)

    @classmethod
    def plot_std(
        cls,
        axis: matplotlib.axes.Axes,
        t: np.ndarray,
        values: np.ndarray,
        *,
        label: str = "+/- std",
        color: str | None = None,
        alpha: float = 0.3,
    ) -> None:
        """
        Plots the standard deviations around the means.

        Args:
            axis (matplotlib.axes.Axes): The axis to plot on.
            t (np.ndarray): The time points.
            values (np.ndarray): The statistics values.
            label (str, optional): The label for the plot. Defaults to "+/- std".
            color (str | None, optional): The color of the plot. Defaults to None.
            alpha (float, optional): The transparency of the fill. Defaults to 0.3.
        """
        var = values[:, MEAN_SQR] - values[:, MEAN] ** 2
        std = np.sqrt(var)
        lower = values[:, MEAN] - std
        upper = values[:, MEAN] + std
        good = np.bitwise_and(~np.isnan(lower), ~np.isnan(upper))

        if color is None:
            axis.fill_between(
                t[good], lower[good], upper[good], alpha=alpha, label=label
            )
        else:
            axis.fill_between(
                t[good], lower[good], upper[good], alpha=alpha, label=label, color=color
            )

    @classmethod
    def plot_mean_std(
        cls,
        axis: matplotlib.axes.Axes,
        t: np.ndarray,
        values: np.ndarray,
        *,
        label: str = "mean",
        label_std: str = "+/- std",
        color: str | None = None,
        alpha: float = 0.3,
    ) -> None:
        """
        Plots the means and standard deviations.

        Args:
            axis (matplotlib.axes.Axes): The axis to plot on.
            t (np.ndarray): The time points.
            values (np.ndarray): The statistics values.
            label (str, optional): The label for the mean plot. Defaults to "mean".
            label_std (str, optional): The label for the std plot. Defaults to "+/- std".
            color (str | None, optional): The color of the plot. Defaults to None.
            alpha (float, optional): The transparency of the fill. Defaults to 0.3.
        """
        cls.plot_mean(axis, t, values, label=label, color=color)
        cls.plot_std(axis, t, values, label=label_std, color=color, alpha=alpha)

    @classmethod
    def plot_mean_range(
        cls,
        axis: matplotlib.axes.Axes,
        t: np.ndarray,
        values: np.ndarray,
        *,
        label: str = "mean",
        label_bounds: str = "min/max",
        color: str | None = None,
        alpha: float = 0.3,
    ) -> None:
        """
        Plots the means and ranges (min/max).

        Args:
            axis (matplotlib.axes.Axes): The axis to plot on.
            t (np.ndarray): The time points.
            values (np.ndarray): The statistics values.
            label (str, optional): The label for the mean plot. Defaults to "mean".
            label_bounds (str, optional): The label for the range plot. Defaults to "min/max".
            color (str | None, optional): The color of the plot. Defaults to None.
            alpha (float, optional): The transparency of the fill. Defaults to 0.3.
        """
        cls.plot_mean(axis, t, values, label=label, color=color)
        cls.plot_range(axis, t, values, label=label_bounds, color=color, alpha=alpha)


################################
## Types of Statistics supported
################################


class Series_Statistics:
    """
    Class to track statistics of a series of values over time intervals.
    """

    __slots__ = [
        "statistics",
        "n_intervals",
        "n_values",
        "min_time_range",
        "max_time_range",
        "min_time",
        "max_time",
        "storage",
        "count",
    ]

    statistics: IntervalStatistics
    n_intervals: int
    n_values: int
    min_time_range: float | None
    max_time_range: float | None

    min_time: float
    max_time: float
    storage: np.ndarray
    count: int

    def __init__(
        self,
        statistics: CountMinMaxMeanVarStd,
        n_intervals: int,
        n_values: int,
        *,
        min_time_range: float | None = None,
        max_time_range: float | None = None,
    ) -> None:
        """
        Initializes the Series_Statistics.

        Args:
            statistics (IntervalStatistics): The statistics object to use.
            n_intervals (int): Number of intervals in the series.
            n_values (int): Number of values to track.
            min_time_range (float | None, optional): The start time of the series. Defaults to None.
            max_time_range (float | None, optional): The end time of the series. Defaults to None.
        """
        self.n_intervals = int(n_intervals)
        if n_intervals < 1:
            raise ValueError("Number of intervals must be, at least, 1")
        self.statistics = statistics
        n_statistics: int = len(statistics)
        self.n_values = n_values
        self.storage: np.ndarray = np.zeros(
            (n_values, n_intervals, n_statistics), dtype=np.float32
        )
        for v in range(n_values):
            for i in range(n_intervals):
                statistics.reset(self.storage[v, i, :])
        self.min_time_range = min_time_range
        self.max_time_range = max_time_range
        self.min_time = +np.inf
        self.max_time = -np.inf
        self.count = 0
        # assert (
        #     len(self.__dict__) == 0
        # ), f"`__slots__` incomplete for `{self.__class__.__name__}`, add {list(self.__dict__.keys())}"

    def __len__(self):
        return self.count

    def double_range(self) -> None:
        """Doubles the time range of the series by merging adjacent intervals."""
        # print(f"double_range: {self.min_time_range}:{self.max_time_range}")
        assert self.min_time_range is not None
        assert self.max_time_range is not None
        old_storage = self.storage.copy()
        for v in range(self.n_values):
            i_new = 0
            # merge consecutive intervals
            for i in range(0, self.n_intervals - 1, 2):
                # print(f"   merge old[{i}+{i+1}] => new[{i_new}]")
                self.statistics.merge_intervals(
                    old_storage[v, i, :],
                    old_storage[v, i + 1, :],
                    self.storage[v, i_new, :],
                )
                i_new += 1
            # clear remaining intervals
            while i_new < self.n_intervals:
                self.statistics.reset(self.storage[v, i_new, :])
                i_new += 1
        self.max_time_range = self.min_time_range + 2 * (
            self.max_time_range - self.min_time_range
        )
        # print(f"            > {self.min_time_range}:{self.max_time_range}")

    def get_index(self, t: float) -> int:
        """
        Calculates the interval index for a given time.

        Args:
            t (float): The time.

        Returns:
            int: The index of the interval.
        """
        if self.min_time_range is None or self.min_time_range == t:
            self.min_time_range = t
            return 0

        if t < self.min_time_range:
            raise ValueError(
                f"Current implementation requires t={t} >= min_time_range={self.min_time_range}"
            )

        if self.max_time_range is None:
            # place current time half-way to end
            self.max_time_range = self.min_time_range + 2 * (t - self.min_time_range)

        # normalize so that max value maps to the "end" of last interval, i.e., n_intervals
        normalized_value = (
            (t - self.min_time_range)
            / (self.max_time_range - self.min_time_range)
            * self.n_intervals
        )
        index = int(normalized_value)

        if index >= self.n_intervals:
            while True:
                self.double_range()

                normalized_value = (
                    (t - self.min_time_range)
                    / (self.max_time_range - self.min_time_range)
                    * self.n_intervals
                )
                index = int(normalized_value)

                if index < self.n_intervals:
                    break

        return index

    def add_value(self, t: float, values: list[float] | np.ndarray) -> None:
        """
        Adds values at a specific time.

        Args:
            t (float): The time at which values are recorded.
            values (list[float] | np.ndarray): The values to record.
        """
        if t < self.min_time:
            self.min_time = t
        if t > self.max_time:
            self.max_time = t
        self.count += 1
        i = self.get_index(t)
        for v, value in enumerate(values):
            # print(f"Adding t={t:.3f}, v={v:3}, value={value}")
            self.statistics.add_value(self.storage[v, i, :], value)
        # assert (
        #     len(self.__dict__) == 0
        # ), f"`__slots__` incomplete for `{self.__class__.__name__}`, add {list(self.__dict__.keys())}"

    def get_series(
        self, value: int, statistic: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the time series for a specific value and statistic.

        Args:
            value (int): The index of the value.
            statistic (int|None): The index of the statistic. When `None` (default), all statistics
                    are returned.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing
                + 1-d vector with time points, and
                + 1-d with the desired statistic (when statistic != None), or
                + matrix with the all the statistics (one per column)

        """
        if self.min_time_range is None or self.max_time_range is None:
            # print(f"min_time_range={self.min_time_range}, max_time_range={self.max_time_range}")
            t = np.ndarray((0,))
            if statistic is None:
                values = np.ndarray((0, self.n_values))
            else:
                values = np.ndarray((0,))
            return (t, values)

        t = (
            self.min_time_range
            + np.arange(self.n_intervals)
            * (self.max_time_range - self.min_time_range)
            / self.n_intervals
        )
        if statistic is None:
            values = self.storage[value, :, :]
        else:
            values = self.storage[value, :, statistic]
        return (t, values)

    def __str__(self) -> str:
        s = f"Series_Statistics(statistics={type(self.statistics).__name__},n_values={self.n_values},n_intervals={self.n_intervals},min_time_range={self.min_time_range},max_time_range={self.max_time_range})\n"
        t0 = self.min_time_range
        for i in range(self.n_intervals):
            if (
                t0 is not None
                and self.min_time_range is not None
                and self.max_time_range is not None
            ):
                t1 = t0 + (self.max_time_range - self.min_time_range) / self.n_intervals
            else:
                t1 = None
            s += f"interval {i:3} [{t0},{t1}):\n"
            t0 = t1
            for v in range(self.n_values):
                s += f"value {v:3}: {self.storage[v,i,:]}\n"
        return s

    def plot(
        self,
        plot_fun: Callable,
        axis: matplotlib.axes.Axes,
        value: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Plots the series using a specific plotting function.

        Args:
            plot_fun (Callable): The function to use for plotting (e.g., CountMinMaxMeanVarStd.plot_mean).
            axis (matplotlib.axes.Axes): The axis to plot on.
            value (int): The index of the value to plot.
            *args: Additional positional arguments for the plotting function.
            **kwargs: Additional keyword arguments for the plotting function.
        """
        if self.min_time == np.inf:  # nothing to plot
            return
        (t, values) = self.get_series(value)
        # values[np.isinf(values)] = np.nan
        # if value == 0:  # loss
        #    print(f"t {t.shape} =")
        #    print(t[:])
        #    print(f"values {values.shape} =")
        #    print(values[:, MEAN])
        plot_fun(axis, t, values, *args, **kwargs)
        if self.min_time == self.max_time:
            axis.set_xlim(self.min_time_range, self.max_time_range)
        else:
            axis.set_xlim(self.min_time, self.max_time)
        # assert (
        #     len(self.__dict__) == 0
        # ), f"`__slots__` incomplete for `{self.__class__.__name__}`, add {list(self.__dict__.keys())}"

    def plot_remove(self, axis: matplotlib.axes.Axes) -> None:
        """
        Removes all lines and collections from the axis.

        Args:
            axis (matplotlib.axes.Axes): The axis to clear.
        """
        # collections needed for scatter and fill_between
        for artist in axis.lines + axis.collections:  # type: ignore
            artist.remove()


if __name__ == "__main__":
    import unittest

    unittest.main()
