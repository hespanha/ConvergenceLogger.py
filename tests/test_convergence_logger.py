"""Unit tests for the convergence_logger package."""

import unittest
from typing import Any
import sys

import matplotlib.pyplot as plt
import numpy as np

from convergence_logger import LoggerStatistics, CountMinMaxMeanVarStd


class TestConvergenceLogger(unittest.TestCase):
    """Tests for the LoggerStatistics class."""

    def test_add_values(self) -> None:
        """Test adding values and retrieving statistical series."""
        n_intervals = 5
        logger = LoggerStatistics(CountMinMaxMeanVarStd(), n_intervals, 2)
        print(logger)

        logger.add_value(1.0, [10, -10.0])
        print(logger)

        # This value should fall into the 3rd interval
        logger.add_value(6.0, [60, -60.0])
        print(logger)

        logger.add_value(2.0, [20, -20.0])
        print(logger)

        logger.add_value(2.0, [20, -20.0])
        print(logger)

        t, cnt = logger.get_series(value=0, statistic=0)
        t, mea = logger.get_series(value=0, statistic=3)
        print("count:", t, cnt)
        print("mean: ", t, mea)

        self.assertTrue(np.allclose(t, [1.0, 3.0, 5.0, 7.0, 9.0]))
        self.assertTrue(np.allclose(cnt, [3.0, 0.0, 1.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(mea[[0, 2]], [(10.0 + 20.0 + 20.0) / 3.0, 60.0]))
        self.assertTrue(np.isnan(mea[[1, 3, 4]]).all)  # 0 counts

        logger.add_value(15.0, [150, -150.0])
        logger.add_value(15.0, [150, -150.0])
        logger.add_value(15.0, [150, -150.0])
        print(logger)

        t, cnt = logger.get_series(value=0, statistic=0)
        t, mea = logger.get_series(value=0, statistic=3)
        print("count:", t, cnt)
        print("mean: ", t, mea)

        self.assertTrue(np.allclose(t, [1.0, 5.0, 9.0, 13.0, 17.0]))
        self.assertTrue(np.allclose(cnt, [3.0, 1.0, 0.0, 3.0, 0.0]))

        self.assertEqual(len(t), n_intervals)
        self.assertEqual(len(cnt), n_intervals)
        self.assertEqual(len(mea), n_intervals)

    def _run_plot_test(self, n_intervals: int) -> None:
        """Helper function to run a single plot test case."""
        print(f"Testing plot with {n_intervals} intervals...")
        n_values = 3
        fig, axes = plt.subplots(3, 1, figsize=(8, 6))
        ax: list[Any] = list(axes)
        for a in ax:
            a.grid(True)
        ax[0].set_ylabel("1st value")
        ax[1].set_ylabel("2nd value")
        ax[2].set_ylabel("2nd and 3rd values")
        ax[2].set_xlabel("iterations")
        # Add a twin axis for the 1st plot to show two different metrics
        ax.append(ax[0].twinx())
        ax[3].set_ylabel("2nd value")

        fig.tight_layout()
        stats = CountMinMaxMeanVarStd()
        logger = LoggerStatistics(stats, n_intervals, n_values)

        for iteration in 100 * np.arange(10 * n_intervals):
            logger.add_value(
                float(iteration),
                [
                    0.9 * (3.0 + iteration * (1.0 + 0.1 * np.random.randn())) ** 2,
                    1.0 * (iteration * (1.0 + 0.1 * np.random.randn())) ** 2,
                    1.2 * (-3.0 + iteration * (1.0 + 0.1 * np.random.randn())) ** 2,
                ],
            )

            # Update the plots every 1000 iterations
            if iteration % 1000 == 0 or iteration == 100 * (10 * n_intervals - 1):
                for a in ax:
                    logger.plot_remove(a)
                logger.plot(
                    stats.plot_mean_std, ax[0], 0, color="C0", label="1st value mean"
                )
                logger.plot(stats.plot_std, ax[3], 1, color="C1", label="2nd value std")
                logger.plot(
                    stats.plot_mean_range, ax[1], 1, color="C1", label="2nd value mean"
                )
                logger.plot(
                    stats.plot_mean, ax[2], 1, color="C1", label="2nd value mean"
                )
                logger.plot(
                    stats.plot_mean_std, ax[2], 2, color="C2", label="3rd value mean"
                )
                # Finalize layout
                ax[0].legend(loc="upper left")
                ax[1].legend(loc="upper left")
                ax[2].legend(loc="upper left")
                ax[3].legend(loc="lower right")

                # force redraw
                fig.canvas.draw()
                fig.canvas.flush_events()
                if "pytest" not in sys.modules:
                    plt.pause(0.0001)

        if "pytest" not in sys.modules:
            fig.savefig("docs/images/example_plot.png", dpi=150)

    def test_plot(self) -> None:
        """Test the plotting functionality of the logger."""
        for n_intervals in [20, 50, 100, 200]:
            self._run_plot_test(n_intervals)
        if "pytest" not in sys.modules:
            plt.show()


if __name__ == "__main__":
    unittest.main()
