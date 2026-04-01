import numpy as np
import matplotlib.pyplot as plt

from convergence_logger import LoggerStatistics, CountMinMaxMeanVarStd

import unittest


class TestConvergenceLogger(unittest.TestCase):
    def test_add_values(self):

        n_intervals = 5
        logger = LoggerStatistics(CountMinMaxMeanVarStd(), n_intervals, 2)
        print(logger)

        logger.add_value(1.0, [10, -10.0])
        print(logger)

        logger.add_value(6.0, [60, -60.0])
        print(logger)

        logger.add_value(2.0, [20, -20.0])
        print(logger)

        logger.add_value(2.0, [20, -20.0])
        print(logger)

        (t, cnt) = logger.get_series(value=0, statistic=0)
        (t, mea) = logger.get_series(value=0, statistic=3)
        print("count:", t, cnt)
        print("mean: ", t, mea)

        self.assertTrue(np.allclose(t, [1.0, 3.0, 5.0, 7.0, 9.0]))
        self.assertTrue(np.allclose(cnt, [3.0, 0.0, 1.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(mea[[0, 2]], [(10 + 20 + 20) / 3.0, 60.0]))
        self.assertTrue(np.isnan(mea[[1, 3, 4]]).all)  # 0 counts

        logger.add_value(15.0, [150, -150.0])
        logger.add_value(15.0, [150, -150.0])
        logger.add_value(15.0, [150, -150.0])
        print(logger)

        (t, cnt) = logger.get_series(value=0, statistic=0)
        (t, mea) = logger.get_series(value=0, statistic=3)
        print("count:", t, cnt)
        print("mean: ", t, mea)

        self.assertTrue(np.allclose(t, [1.0, 5.0, 9.0, 13.0, 17.0]))
        self.assertTrue(np.allclose(cnt, [3.0, 1.0, 0.0, 3.0, 0.0]))

        self.assertEqual(len(t), n_intervals)
        self.assertEqual(len(cnt), n_intervals)
        self.assertEqual(len(mea), n_intervals)

    def test_plot(self):

        for n_intervals in [20, 50, 100, 200]:
            n_values = 3
            fig, ax = plt.subplots(3, 1)
            for a in ax:
                a.grid(True)
            ax = np.append(ax, ax[2].twinx())
            ax[2].set_ylabel("1st value")
            ax[3].set_ylabel("2nd value")
            stats = CountMinMaxMeanVarStd()
            logger = LoggerStatistics(stats, n_intervals, n_values)
            fig.show()
            # print(logger)

            for t in 100 * np.arange(10 * n_intervals):
                logger.add_value(
                    float(t),
                    [
                        0.9 * (3.0 + t * (1.0 + 0.1 * np.random.randn())) ** 2,
                        1.0 * (t * (1.0 + 0.1 * np.random.randn())) ** 2,
                        1.2 * (-3.0 + t * (1.0 + 0.1 * np.random.randn())) ** 2,
                    ],
                )
            # print(logger)
            for a in ax:
                logger.plot_remove(a)
            logger.plot(stats.plot_mean_range, ax[0], 0)
            logger.plot(stats.plot_std, ax[0], 1)
            logger.plot(stats.plot_mean_std, ax[1], 1)
            logger.plot(stats.plot_mean, ax[2], 2, color="C0")
            logger.plot(stats.plot_mean, ax[3], 1, color="C1")
            ax[0].legend()
            ax[1].legend()
            ax[2].legend(loc="upper left")
            ax[3].legend(loc="lower right")
            plt.tight_layout()
        plt.show(block=True)


test = TestConvergenceLogger()
test.test_add_values()
test.test_plot()
