"""Test different statistical tests methods by repeating them multiple times
    and calculating percentage of errors
"""
from scipy.stats import (wasserstein_distance,
                         energy_distance)
import examples as ex
from utils import (mean_distance,
                   kolmogorov_smirnov,
                   epps_singleton,
                   cramer_distance,
                   alexander_distance,
                   ttest_dist,
                   calculate_stats_distances)


# test for different distances
distance_metrics = [wasserstein_distance,
                    energy_distance,
                    mean_distance,
                    kolmogorov_smirnov,
                    epps_singleton,
                    cramer_distance,
                    alexander_distance,
                    ttest_dist]
NUM_OF_ITERATIONS = 5000
for method in distance_metrics:
    num_total_warnings = 0
    for i in range(NUM_OF_ITERATIONS):
        girls_train = ex.girls_height_distribution(80, 101)
        boys_train = ex.boys_height_distribution(80, 101)
        boys_test = ex.boys_height_distribution(40, 51)
        girls_test = ex.girls_height_distribution(40, 51)
        num_warnings, _ = calculate_stats_distances(
            boys_train, girls_train, boys_test, girls_test, method=method)
        num_total_warnings += num_warnings
    print(f"num_warnings for {method.__name__}: ",
          num_total_warnings)
    print(f"total_error_percentage: \
    {num_total_warnings / (2*(NUM_OF_ITERATIONS))*100:.2f}%")
