# convergence_logger.py

A Python package to monitor and visualize the convergence of machine learning algorithms by tracking statistics over time.

## Features

- **Time Compression**: Automatically aggregates data into a fixed number of time intervals to manage memory.
- **Statistical Tracking**: Computes count, min, max, mean, variance, and standard deviation on the fly.
- **Visualization**: Built-in methods for easy plotting with `matplotlib`.

## Installation

Create a local environment and install `convergence_logger` and its dependencies using `pip`:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install git+https://github.com/hespanha/ConvergenceLogger.py
    ```

## Usage Example

    ```python
    from convergence_logger import LoggerStatistics, CountMinMaxMeanVarStd
    import matplotlib.pyplot as plt
    import numpy as np

    # Logger configuration
    # The logger will track 3 different values/metrics
    n_values = 3

    # The logger will compress time into a maximum of 200 time intervals
    n_intervals = 200

    # Configure the logger to track: count, min, max, mean, variance, and std-dev
    stats = CountMinMaxMeanVarStd()

    # Create the logger instance
    logger = LoggerStatistics(stats, n_intervals, n_values)

    # Create figure and axes for plots
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    for a in ax:
        a.grid(True)

    # Optional: Add a twin axis for the last plot
    ax = np.append(ax, ax[2].twinx())
    ax[2].set_ylabel("Metric A")
    ax[3].set_ylabel("Metric B")

    ## Add values to the logger (simulated data)
    for t in np.linspace(0, 10, 2000):
        logger.add_value(
            float(t),
            [
                0.9 * (3.0 + t * (1.0 + 0.1 * np.random.randn())) ** 2,
                1.0 * (t * (1.0 + 0.1 * np.random.randn())) ** 2,
                1.2 * (-3.0 + t * (1.0 + 0.1 * np.random.randn())) ** 2,
            ],
        )

    ## Update the plots
    # Remove previous data from all axes
    for a in ax:
        logger.plot_remove(a)

    # Add specific statistical plots to the axes
    logger.plot(stats.plot_mean_range, ax[0], 0)
    logger.plot(stats.plot_std, ax[0], 1)
    logger.plot(stats.plot_mean_std, ax[1], 1)
    logger.plot(stats.plot_mean, ax[2], 2, color="C0", label="Value 2")
    logger.plot(stats.plot_mean, ax[3], 1, color="C1", label="Value 1")

    # Finalize layout
    ax[0].legend()
    ax[1].legend()
    ax[2].legend(loc="upper left")
    ax[3].legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    ```
