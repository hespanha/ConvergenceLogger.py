from importlib.metadata import PackageNotFoundError, version

__all__ = ["LoggerStatistics", "CountMinMaxMeanVarStd"]

try:
    __version__ = version("convergence_logger")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

from .convergence_logger import LoggerStatistics, CountMinMaxMeanVarStd
