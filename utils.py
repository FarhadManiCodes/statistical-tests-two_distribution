"""Produce example/toy distributions for evaluating the tests"""

from typing import  Tuple
import numpy as np

def make_same_size(a:np.ndarray, b :np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    "make the samples the same size"
    min_size =  min(len(a), len(b))
    s_a = np.random.choice(a, min_size,replace=False)
    s_b = np.random.choice(b, min_size,replace=False)
    return s_a, s_b