import examples as ex
from random import randrange
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from scipy.stats import (wasserstein_distance,
                         energy_distance)


# test for different distances
distance_metrics = [wasserstein_distance, 
                    energy_distance,
                     mean_distance, 
                     ks,
                     epps_singleton,
                     cramer_distance]
num_total_warnings = 0
num_of_iterations = 10000
for method in distance_metrics:
    for i in range(num_of_iterations):
        girls_train = ex.girls_height_distribution(80, 101)
        boys_train = ex.boys_height_distribution(80, 101)
        boys_test = ex.boys_height_distribution(40, 51)
        girls_test = ex.girls_height_distribution(40, 51)
        num_warnings, _ = calculate_stats_distances(
            boys_train, girls_train, boys_test, girls_test, method=method)
        num_total_warnings += num_warnings
    print(f"num_warnings for {method.__name__}: ", num_total_warnings)
    print(
        f"total_error_percentage: {num_total_warnings / (2*(num_of_iterations))*100:.2f}%")
