"""Produce example/toy distributions for evaluating the tests"""

from typing import List
import numpy as np
from random import randrange

def make_test_dist_norm(list_mu:List[float],
                        list_sigma:List[float],
                        size_list:List[int]) -> np.ndarray:
    """
    to make distributions from normal distribution

    this function create a combination of several normal distributions with different
    size, mean, and standard deviation.
    
    Parameters
    ----------
    list_mu: List[float]
        list on means to be used.
    list_sigma:List[float]
        list of standard deviations to be used.
    size_list: List[int]
        size of each normal distribution.

    Returns
    ---------
    np.ndarray
        new constructed distribution.
    """
    s = []
    for mu, sigma, size in zip(list_mu, list_sigma, size_list):
        s.append(np.random.normal(mu, sigma,size))
    return  np.concatenate(s,axis=0)


def boys_height_distribution(min:int = 1, max:int = 100) -> np.ndarray:
    """
    create distribution of hight for boys
    
    Parameters
    -----------
    min:int
        minimum size from each month distribution 
    end:int
    maximum size from each month distribution

    Returns
    -----------
    np.ndarray
    example distribution of height for boys
    """
    # mean of height for boys age 6.5--7.5
    MU_Bs = [118.8700, 119.3508, 119.8303, 120.3085,120.7853, 121.2604,
            121.7338, 122.2053,122.6750, 123.1429, 123.6092, 124.0736] 
    # standard deviation for boys height age 6.5--7.5
    SIGMA_Bs = [5.1055, 5.1357, 5.1659, 5.1949, 5.2252, 5.2554, 
                5.2857, 5.3159, 5.3462 ,5.3764, 5.4067,  5.4369] 
    
    size_list = [randrange(min, max) for i in range(12)]
    return make_test_dist_norm(MU_Bs,SIGMA_Bs,size_list)


def girls_height_distribution(min:int = 1, max:int = 100):
    """
    create distribution of hight for girls
    
    Parameters
    -----------
    min:int
        minimum size from each month distribution 
    end:int
        maximum size from each month distribution

    Returns
    -----------
    np.ndarray
    
    example distribution of height for girls
    
    """
    # mean of height for girls age 6.5--7.5
    MU_Gs = [117.9769, 118.4489, 118.9208, 119.3926, 119.8648, 120.3374,
            120.8105, 121.2843, 121.7587, 122.2338, 122.7098, 123.1868] 
    # standard deviation for girls height age 6.5--7.5
    SIGMA_Gs = [5.2960, 5.3243, 5.3538, 5.3822, 5.4107, 5.4393,
                5.4667, 5.4954, 5.5230, 5.5519, 5.5796, 5.6062] 
    size_list = [randrange(min, max) for i in range(12)]
    return make_test_dist_norm(MU_Gs,SIGMA_Gs,size_list)