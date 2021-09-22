"""Produce example/toy distributions for evaluating the tests"""

from typing import  Callable, Dict, Tuple
import numpy as np
from scipy.stats import (ks_2samp,
                         cramervonmises_2samp,
                         epps_singleton_2samp,
                         anderson_ksamp,
                         alexandergovern,
                         ttest_rel)
import warnings


def make_same_size(a:np.ndarray, b :np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    "make the samples the same size"
    min_size =  min(len(a), len(b))
    s_a = np.random.choice(a, min_size,replace=False)
    s_b = np.random.choice(b, min_size,replace=False)
    return s_a, s_b

def calculate_stats_distances(a_train:np.ndarray,b_train:np.ndarray,
                              a_test:np.ndarray, b_test:np.ndarray,
                              method:Callable) -> Tuple[int,Dict]:
    b_train_test = method(b_train,b_test)
    a_train_test = method(a_train,a_test)
    a_train_b_test = method(a_train,b_test)
    b_train_a_test = method(b_train,a_test)
    num_warnings = 0
    if b_train_a_test < a_train_test:
        warnings.warn('could not specify correct class for the fist distribution')
        num_warnings += 1
    if a_train_b_test < b_train_test:
        warnings.warn('could not specify correct class for the fist distribution')
        num_warnings+=1
    return num_warnings, {"a_train_test":a_train_test,
            "b_train_test":b_train_test,
            "a_train_b_test":a_train_b_test,
            "b_train_a_test":b_train_a_test}

def mean_distance(a:np.ndarray, b:np.ndarray) -> float:
    """calculate the distance between the distributions based on their expected values"""
    return abs(a.mean()-b.mean())

def epps_singleton(a:np.ndarray, b:np.ndarray) -> float:
    """
    calculate the distance between the distributions 
    based on Epps-Singleton
    
    """
    return  epps_singleton_2samp(a,b).statistic


def ks(a:np.ndarray, b:np.ndarray) -> float:
    """calculate the distance between the distributions
    based on Kolmogorov–Smirnov test
    
    """
    return  1.0 - ks_2samp(a,b).pvalue

def cramer_distance(a:np.ndarray, b:np.ndarray) -> float:
    """
    calculate the distance between the distributions 
    based on  Cramér-von Mises test
    
    """
    return  1.0 - cramervonmises_2samp(a,b).pvalue

def alexander_distance(a:np.ndarray, b:np.ndarray) -> float:
    """
    calculate the distance between the distributions 
    based on Alexander Govern test.
    
    """
    return 1.0 - alexandergovern(a,b).pvalue


def ttest_dist(a:np.ndarray, b:np.ndarray) -> float:
    """
    calculate the distance between the distributions 
    based on ttest.
    
    """
    a_s, b_s = make_same_size(a,b)
    return 1-ttest_rel(a_s,b_s).pvalue