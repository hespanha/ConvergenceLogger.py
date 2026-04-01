# convergence_logger.py

A python package to monitor the convergence of ML algorithms.

## Installation

Create a local environment and install track_results and its dependencies using `pip`

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install git+https://github.com/hespanha/ConvergenceLogger.py
    ```

## Exmaple if how to use the package

    ```python
    from convergence_logger import LoggerStatistics, CountMinMaxMeanVarStd
    import matplotlib.pyplot as plt
    import numpy as np

    ## Logger parameters
    # logger will keep track of 3 values
    n_values = 3
    # logger will compress time into a maximum of 200 time intervals    
    n_intervals=200:
    # logger will keep track of:
    #      points (count), min, max, mean, var and std-dev 
    # of each value in each interval
    stats = CountMinMaxMeanVarStd()

    ## Create logger
    logger = LoggerStatistics(stats, n_intervals, n_values)

    ## Create figure and axis for plots
    fig, ax = plt.subplots(3, 1)
    for a in ax:
        a.grid(True)
    ax = np.append(ax, ax[2].twinx())
    ax[2].set_ylabel("1st value")
    ax[3].set_ylabel("2nd value")
    fig.show()

    ## Add values to logger
    for t in 100 * np.arange(10 * n_intervals):
        logger.add_value(
            float(t),
            [
                0.9 * (3.0 + t * (1.0 + 0.1 * np.random.randn())) ** 2,
                1.0 * (t * (1.0 + 0.1 * np.random.randn())) ** 2,
                1.2 * (-3.0 + t * (1.0 + 0.1 * np.random.randn())) ** 2,
            ],
        )

    ## Remove previous data from all axis
    for a in ax:
        logger.plot_remove(a)
    ## Add plots to the axis
    logger.plot(stats.plot_mean_range, ax[0], 0)
    logger.plot(stats.plot_std, ax[0], 1)
    logger.plot(stats.plot_mean_std, ax[1], 1)
    logger.plot(stats.plot_mean, ax[2], 2, color="C0")
    logger.plot(stats.plot_mean, ax[3], 1, color="C1")
    ## Add legends
    ax[0].legend()
    ax[1].legend()
    ax[2].legend(loc="upper left")
    ax[3].legend(loc="lower right")
    plt.tight_layout()
    ```
