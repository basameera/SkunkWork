"""
SkunkWork Utils
===============

"""

import warnings
import numpy as np

__all__ = ["calc_rmse", "rmse"]


def calc_rmse(target, prediction):
    return np.sqrt(np.mean((prediction-target)**2, axis=0))


def rmse(target, prediction):
    warnings.warn('rmse() will be deprected in the future. Please switch to using calc_rmse() instead.')
    return np.sqrt(np.mean((prediction-target)**2, axis=0))


# run
if __name__ == '__main__':
    pass
