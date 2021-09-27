"""Produce example/toy distributions for evaluating the tests"""

from typing import Callable, Dict, Tuple
import warnings
import numpy as np
from scipy.stats import (ks_2samp,
                         cramervonmises_2samp,
                         epps_singleton_2samp,
                         alexandergovern,
                         ttest_rel)


def make_same_size(first_distribution: np.ndarray,
                   second_distribution: np.ndarray) \
                   -> Tuple[np.ndarray, np.ndarray]:
    "make the samples the same size"
    min_size = min(len(first_distribution), len(second_distribution))
    s_a = np.random.choice(first_distribution, min_size, replace=False)
    s_b = np.random.choice(second_distribution, min_size, replace=False)
    return s_a, s_b


def calculate_stats_distances(a_train: np.ndarray, b_train: np.ndarray,
                              a_test: np.ndarray, b_test: np.ndarray,
                              method: Callable) -> Tuple[int, Dict]:
    """Calculate the statistical distance between distributions

    this function calculate the distance between the distributions
    and will check if the prodiction was correct or not

    Parameters
    ----------
    a_train,b_train:
        train distributions

    a_test,b_test:
        test distributions

    Returns
    --------
    num_warnings:int
        number of warnings and error if faced during checking the test

    Dict
        Dictionary of percentage of failure
    """
    b_train_test = method(b_train, b_test)
    a_train_test = method(a_train, a_test)
    a_train_b_test = method(a_train, b_test)
    b_train_a_test = method(b_train, a_test)
    num_warnings = 0
    if b_train_a_test < a_train_test:
        warnings.warn("""could not specify correct class for
                            the fist distribution""")
        num_warnings += 1
    if a_train_b_test < b_train_test:
        warnings.warn("""could not specify correct class for
                            the second distribution""")
        num_warnings += 1
    return num_warnings, {"a_train_test": a_train_test,
                          "b_train_test": b_train_test,
                          "a_train_b_test": a_train_b_test,
                          "b_train_a_test": b_train_a_test}


def mean_distance(first_distribution: np.ndarray,
                  second_distribution: np.ndarray) -> float:
    """calculate the distance between the distributions
        based on their expected values
    """
    return abs(first_distribution.mean()-second_distribution.mean())


def epps_singleton(first_distribution: np.ndarray,
                   second_distribution: np.ndarray) -> float:
    """
    calculate the distance between the distributions
    based on Epps-Singleton

    """
    return epps_singleton_2samp(first_distribution,
                                second_distribution).statistic


def kolmogorov_smirnov(first_distribution: np.ndarray,
                       second_distribution: np.ndarray) -> float:
    """calculate the distance between the distributions
    based on Kolmogorov–Smirnov test

    """
    return 1.0 - ks_2samp(first_distribution,
                          second_distribution).pvalue


def cramer_distance(first_distribution: np.ndarray,
                    second_distribution: np.ndarray) -> float:
    """
    calculate the distance between the distributions
    based on  Cramér-von Mises test

    """
    return 1.0 - cramervonmises_2samp(first_distribution,
                                      second_distribution).pvalue


def alexander_distance(first_distribution: np.ndarray,
                       second_distribution: np.ndarray) -> float:
    """
    calculate the distance between the distributions
    based on Alexander Govern test.

    """
    return 1.0 - alexandergovern(first_distribution,
                                 second_distribution).pvalue


def ttest_dist(first_distribution: np.ndarray,
               second_distribution: np.ndarray) -> float:
    """
    calculate the distance between the distributions
    based on ttest.

    """
    a_s, b_s = make_same_size(first_distribution, second_distribution)
    return 1-ttest_rel(a_s, b_s).pvalue
