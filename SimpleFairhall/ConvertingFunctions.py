from brian2 import *
import numpy as np
from typing import Tuple

from opencvtry import cvWriter


######################## Converting functions ##########################
def reduce_matrix(my_matrix: np.array, divisor: int) -> np.array:
    """
    Reduces a matrix (takes 1/divcreate_trajectory_matrixisor of its elements).
    :param my_matrix: A matrix
    :type my_matrix: np.array
    :param divisor: A divisor, determines what proportion of the matrix we take
    :type divisor: int
    :return: A reduced matrix
    :rtype: np.array
    """
    rows = int(my_matrix.shape[0] / divisor)
    cols = int(my_matrix.shape[1] / divisor)
    result = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            result[i, j] = my_matrix[i * divisor, j * divisor]
    return result


def convert_matrix_pos_to_indices(number: int, my_matrix: np.array, rows: int, cols: int) -> np.array:
    """
    Takes a matrix and Will give the returns a matrix with column[i]= 1 if the element on the ith position in the
    matrix is greater than 1. Or 0 otherwise.
    :param number: Represents the number of pyramidal cells.
    :type number: int
    :param my_matrix: A matrix that has positive values where the corresponding cell (each cell corresponds to
    one entry in the matrix) is in the trajectory. 0 otherwise.
    :type my_matrix:np.array
    :param rows: number of rows
    :type rows:int
    :param cols: number of columns
    :type cols:int
    :return:Matrix with column[i] is 1 if i is in the trajectory, 0 otherwise
    :rtype:np.array
    """
    # my_indices = []
    result = np.zeros((number, rows * cols))
    for i in range(rows * cols):
        if my_matrix[i % rows, i // rows] > 0:
            # my_indices.append(i)
            result[:, i] = np.ones(number)
    return result


def convert_list_to_threshold(my_list: list, threshold_trajectory: float, threshold_out: float) -> list:
    """
    Creates a new list, for positive elements it takes the value threshold_trajectory, otherwise threshold_out.
    :param my_list: Any list of floats. Strictly positive elements represent the ones inside the trajectory,
    negative (or zero) the ones outside the trajectory.
    :type my_list: list
    :param threshold_trajectory: the value the elements inside the trajectory take.
    :type threshold_trajectory: float
    :param threshold_out:the value the elements outside the trajectory take.
    :type threshold_out: float
    :return: A new list, with two possible values: threshold_trajectory or threshold_out
    :rtype: list
    """
    new_list = [threshold_out] * len(my_list)
    for i in range(len(my_list)):
        if my_list[i] > 0:
            new_list[i] = threshold_trajectory
    return new_list


def convert_matrix_to_source_target(my_matrix: np.array) -> Tuple[np.array, np.array]:
    """
    Converts a matrix into two arrays that give the indices of the non zero elements.
    (Source[i], Target[i]) gives the position of a non zero element.
    :param my_matrix: A matrix of zero or non zero elements.
    :type my_matrix: np.array
    :return: Two arrays of indices that indicate the non zero indices of the matrix
    :rtype: Tuple[np.array, np.array]
    """

    # Return the indices of the elements that are non-zero.
    sources, targets = my_matrix.nonzero()
    sources = np.array(sources)
    targets = np.array(targets)
    return sources, targets


def convert_neg_matrix_to_source_target(my_matrix: np.array) -> Tuple[np.array, np.array]:
    """
    Converts a matrix into two arrays that give the indices of the non zero elements.
    (Source[i], Target[i]) gives the position of a non zero element.
    :param my_matrix: A matrix of zero or non zero elements.
    :type my_matrix: np.array
    :return: Two arrays of indices that indicate the non zero indices of the matrix
    :rtype: Tuple[np.array, np.array]
    """

    # Return the indices of the elements that are non-zero.
    I = np.ones(my_matrix.shape)
    neg_matrix = I - my_matrix
    sources, targets = neg_matrix.nonzero()
    sources = np.array(sources)
    targets = np.array(targets)
    return sources, targets


def convert_to_weight_matrix(source: int, my_list: list, w_group1: float, w_group2: float) -> np.array:
    """
    Returns a matrix of size (source x len(my_list)).
    With row[i] = w_group1 if my_list[i] > 0
                = w_group2 if my_list[i] <= 0
    :param source: Number of rows we want for the matrix.
    :type source: int
    :param my_list: Any list
    :type my_list: list
    :param w_group1: The coef for the rows where my_list[i] > 0
    :type w_group1: float
    :param w_group2: The coef for the rows where my_list[i] <= 0
    :type w_group2: float
    :return: A matrix of size (source x len(my_list))
    :rtype: np.array
    """
    my_matrix = np.zeros([source, len(my_list)])
    for i in range(len(my_list)):
        if my_list[i] > 0:
            my_matrix[:, i] = w_group1
        else:
            my_matrix[:, i] = w_group2

    return my_matrix


def convert_to_movie(list_frames: list, height: int, width: int, filePathName: str, num_frames: int = 10):
    """
    Given a list of frames, converts it into a movie.
    :param list_frames: List of frames (matrices).
    :type list_frames: list of np.array
    :param height: height of a frame
    :type height: int
    :param width: width of a frame
    :type width: int
    :param filePathName: the path where to save the movie
    :type filePathName: str
    :param num_frames: number of frames per image.
    :type num_frames: int

    """
    base_matrix = np.zeros((width, height, 4))
    base_matrix[:, :, 3] = 255 * np.ones((width, height))

    FrameDim = (width, height)
    with cvWriter(filePathName, FrameDim) as vidwriter:
        for frame in list_frames:
            base_matrix[:, :, 1] = frame
            for _ in range(num_frames):
                vidwriter.write(base_matrix)
    return base_matrix
