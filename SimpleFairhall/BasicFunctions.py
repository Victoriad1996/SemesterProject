from brian2 import*
import numpy as np
from typing import Tuple

######################### Basic Functions #########################

# Returns a list of indices. Will be useful for plotting.
def some_indices(neuron_idx : int) -> list:
    """
    Creates a list of indices.

    :param neuron_idx: An index
    :type neuron_idx: int
    :return: List of indices
    :rtype: list of int
    """
    neuron_idx2 = neuron_idx - 1
    neuron_idx3 = 16
    new_idx = 0
    rows = 20
    cols = 30

    for j in range(rows * cols):
        if (j//rows == 10 and j % rows == 5):
            new_idx = j

    my_indices = [neuron_idx, neuron_idx2, neuron_idx3, new_idx]
    return my_indices

def distance(PC : NeuronGroup, neuron_idx:int, j:int) -> float:
    """

    :param PC: Pyramidal cells
    :type PC: NeuronGroup
    :param neuron_idx: One neuron index
    :type neuron_idx: int
    :param j: One neuron index
    :type j: int
    :return: The l2 distance between the cells indexed by neuron_idx and j
    :rtype: float
    """

    result = ((PC.x[neuron_idx] - PC.x[j]) ** 2 + (PC.y[neuron_idx] - PC.y[j]) ** 2) / (meter)**2

    return result

def list_distance(PC : NeuronGroup, neuron_idx : int) -> list:
    """
    Computes the rescaled distance (between 0 and 1) of all pyramidal cells to the neuron_idx.
    Uses the function distance(PC, neuron_idx, j).

    :param PC: Pyramidal cells
    :type PC: NeuronGroup
    :param neuron_idx: One neuron index
    :type neuron_idx: int
    :return: Rescaled distance (between 0 and 1) between the cell of neuro-idx and the rest of the other cells.
    :rtype: list of float
    """
    n = len(PC.x)
    list_dist = [distance(PC, neuron_idx, j) for j in range(n)]
    max_ = max(list_dist)
    result = [distance(PC, neuron_idx, i)/max_ for i in range(n)]
    return result

# Checks if a list is only nans
def is_list_nan(mylist:list) -> bool:
    """
    Checks if the list contains only nan.
    :param mylist: any list
    :type mylist: list
    :return: True if the list contains only nans, False if there is an element that is not a nan.
    :rtype: bool
    """
    result = True
    for i in mylist:
        if not np.isnan(i):
            result = False
    return result

def is_list_inf(mylist : list) -> bool:
    """
    Checks if the list contains only inf.
    :param mylist: any list
    :type mylist: list
    :return: True if the list contains only inf, False if there is an element that is not a inf.
    :rtype: bool
    """
    result = True
    for i in mylist:
        if not np.isinf(i):
            result = False
    return result

# Normalises a list of positive values to a list of values between 0 and 1.
# It helps for plotting spikes: for "alpha" needs values between 0 and 1.
def normalize(l : list, lower: float =0, upper : float =1) -> Tuple[list, int, int]:
    """
    Linear normalisation of the list into a new list contained in [lower, upper]
    :param l: any list of float, inf or nan
    :type l: list
    :param lower: the lower element of the new list
    :type lower: float
    :param upper: the greater element of the new list
    :type upper: float
    :return: Normalised list, argmin, argmax
    :rtype: Tuple[list, int, int]
    """
    # Takes out the "inf" values in order to compute the max, min and length of all given values.
    if lower>=upper:
        raise ValueError("Lower key must be strictly smaller than upper key.")
    if is_list_nan(l):
        return [], 0, 0

    norm_l = [i for i in l if not np.isnan(i)]

    min_ = np.min(norm_l)
    max_ = np.max(norm_l)
    argmin_ = list(np.where(l == min_))[0][0]
    argmax_ = list(np.where(l == max_))[0][0]

    length = max_ - min_
    result = []

    a = upper - lower
    b= lower


    if length > 0:
        for i in l:

            if not np.isnan(i):
                result.append(a * (i-min_)/length + b)
            # if they are some inf values, return them.
            else:
                result.append(i)

    # If all values are the same, then just returns 1s.
    else:
        for i in l:
            if not np.isnan(i):
                result.append(lower)
            else:
                result.append(i)

    # Also returns argmin_, argmax_. It helps for knowing who spikes first and last (among the one that actually spike).
    return result, argmin_, argmax_

# Plots the distribution of any list.
def plot_distrib(my_list:list, name_title:str, my_xlabel : str = None):
    """
    Plots the histogram distribution of a list
    :param my_list: Any list
    :type my_list: list
    :param name_title: The title of my plot
    :type name_title: str
    :param my_xlabel: Label of x (the measure)
    :type my_xlabel: str
    """
    hist(my_list, color = 'blue', edgecolor = 'black',
         bins = int(180/5))
    title("Distribution of " + str(name_title))
    if not my_xlabel is None:
        xlabel(my_xlabel)
    show()

def sgmd(x:float) -> float:
    """
    Sigmoid (logistic) function.
    :param x: number
    :type x: float
    :return: sigmoid transformation of x
    :rtype: float
    """

    return 1 / (1 + np.exp(-x))

