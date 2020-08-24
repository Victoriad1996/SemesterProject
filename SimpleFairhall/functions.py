from brian2 import exp, meter, ms, NeuronGroup, Synapses, SpikeMonitor, start_scope
#from Fairhall import FairhallModel
from Model import Model
#from ThresholdModel import ThresholdModel
from matplotlib.pyplot import*
from numpy import*
from math import isnan
from opencvtry import cvWriter
import matplotlib.pyplot as plt
import copy
from IPython.display import Video
from time import time

from typing import Tuple, Union

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
        if not isnan(i):
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


######################## Converting functions ##########################""
def reduce_matrix(my_matrix : np.array, divisor: int) -> np.array:
    """
    Reduces a matrix (takes 1/divisor of its elements).
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

            result[i,j] = my_matrix[i * divisor, j * divisor]
    return result


def convert_matrix_pos_to_indices(number:int, my_matrix:np.array, rows : int, cols:int) -> np.array:
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
        if my_matrix[i%rows, i//rows] > 0:
            #my_indices.append(i)
            result[:,i] = np.ones(number)
    return result


def convert_list_to_threshold(my_list:list, threshold_trajectory:float, threshold_out:float) -> list:
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
        if my_list[i]>0:
            new_list[i] = threshold_trajectory
    return new_list

def convert_matrix_to_source_target(my_matrix : np.array) -> Tuple[np.array, np.array]:
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

def convert_neg_matrix_to_source_target(my_matrix : np.array) -> Tuple[np.array, np.array]:
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

def convert_to_weight_matrix(source:int, my_list:list, w_group1:float, w_group2:float) -> np.array:
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
        if my_list[i]>0:
            my_matrix[:,i] = w_group1
        else:
            my_matrix[:,i] = w_group2

    return my_matrix

def convert_to_movie(list_frames:list, height:int, width:int, filePathName:str, num_frames:int = 10):
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
            base_matrix[:,:,1] = frame
            for _ in range(num_frames):
                vidwriter.write(base_matrix)
    return base_matrix


###### Plot Network Structure ######

def plot_different_distances(modelClass):
    """
    Plot the distribution of different distances. Helps for construction of the network.
    :param modelClass: A Network model that has PC elements.
    :type modelClass: Either ThresholdModel or FairhallModel
    """

    n = modelClass.p['rows'] * modelClass.p['cols']
    list1 = []
    list2 = []
    for j in range(n):
        list1.append(
            2 - 2 * exp(-((modelClass.PC.x[50] - modelClass.PC.x[j]) ** 2 + (modelClass.PC.y[50] - modelClass.PC.y[j]) ** 2) / ((60 * meter) ** 2)))
        list2.append(
            (((modelClass.PC.x[50] - modelClass.PC.x[j]) ** 2 + (modelClass.PC.y[50] - modelClass.PC.y[j]) ** 2) / ((2 * 15 * meter) ** 2)) / 10)

    plot_distrib(list1, "2 - 2 * exp(-||n1 - n2||_2)")
    plot_distrib(list2, "||n1 - n2||_2")


# Visualise connectivity of a group of synapses given in parameter. Represented by vertices and edges.
def visualise_connectivity(S : Synapses, mylegend:str):
    """
    Visualise the connectivity of the synapses. Between the source neurons and target neurons.
    :param S: Synapses
    :type S: Synapses
    :param mylegend: Legend of the plot
    :type mylegend: str
    """
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
        #, linewidth=S.w[i, j][0]
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index', rotation='vertical')
    suptitle(mylegend)
    show()

def plot_all_connectivity(modelClass, neuron_idx : int =None):
    """
    Plots all the different connections.
    :param modelClass: Network Model
    :type modelClass: Either ThresholdModel or Fairhall Model
    :param neuron_idx: The index of a neuron.
    :type neuron_idx: int or None
    """

    dict_synapses = modelClass.get_dict_synapses()
    list_synapses = list(dict_synapses['PC'].values())
    list_title = list(dict_synapses['PC'].keys())
    plot_connectivity(modelClass, list_synapses, list_title, neuron_idx)

    return dict_synapses

# If gives a neuron_idx, then plots all the pyramidal cells it's connected to by the synapses S.
# Otherwise plots all neurons in the synapses S.
def plot_connectivity(modelClass, list_synapses:list, my_title:list, neuron_idx = None):
    """
    Plots
    If neuron_idx is int : the plots of the connectivity created by the synapses S inside the pyramidal cells.
    If neuron_idx is none :
    :param modelClass: Network Model
    :type modelClass: Either ThresholdModel or Fairhall Model
    :param list_synapses: List of Synpases
    :type list_synapses: list of Synapses
    :param my_title: List of title of plots
    :type my_title:list[str]
    :param neuron_idx: The index of a neuron.
    :type neuron_idx: int or None
    """
    color = 'b'
    # Plots all the pyramidal cells it's connected to by the synapses S.
    j = 0
    if neuron_idx == None:
        for i in range(modelClass.p['rows']*modelClass.p['cols']):
            plot(modelClass.PC.x[i] / meter, modelClass.PC.y[i] / meter, color + '.', alpha = 0.8)
        for S in list_synapses:
            plot(modelClass.PC.x[0] / meter, modelClass.PC.y[0] / meter, 'k' + '.', alpha=0.)
            plot(modelClass.PC.x[599] / meter, modelClass.PC.y[599] / meter, 'k' + '.', alpha=0.)
            plot(S.x / meter, S.y / meter, color + '.')

            xlabel("meter")
            ylabel("meter", rotation='vertical')
            title("Connectivity between " + str(my_title[j]))
            show()
            j += 1



    # Plots all neurons in the synapses S.
    else:
        # The neurons connected by synapses S to neuron_idx
        for S in list_synapses:
            plot(modelClass.PC.x[0] / meter, modelClass.PC.y[0] / meter, 'k' + '.', alpha=0.)
            plot(modelClass.PC.x[599] / meter, modelClass.PC.y[599] / meter, 'k' + '.', alpha=0.)
            plot(S.x[neuron_idx,:] / meter, S.y[neuron_idx,:] / meter, 'r' + '.')
            xlabel("meter")
            ylabel("meter", rotation='vertical')
            title("Connectivity between " + str(my_title[j]))
            show()
            j += 1

        # Plots the neuron_idx.
        plot(modelClass.PC.x[neuron_idx] / meter, modelClass.PC.y[neuron_idx] / meter, color + '.', alpha=0.8, label="Main neuron")

# Plot, with gradient color, the distance of each neurons to the "neuron_idx" neuron given in parameters.
def plot_distance(modelClass):
    """
    Plots the distance of all the pyramidal cell to a the neuron_idx one.
    neuron_idx is an attribute of the class Model.
    :param modelClass: A network model
    :type modelClass: Either ThresholdModel or FairhallModel
    """

    # Computes the distance of all PC cells to the neuron_idx and returns a list.
    my_list = list_distance(modelClass.PC, modelClass.neuron_idx)
    # Plots the distribution of this list of distances.
    plot_distrib(my_list, "distances")

    # Marks the neuron_idx
    plot(modelClass.PC.x[modelClass.neuron_idx] / meter, modelClass.PC.y[modelClass.neuron_idx] / meter, 'o',
                mfc='none', label=str(modelClass.neuron_idx))
    color = 'b'

    # The alpha allows to "grade" the distance to the neuron cell.
    for i in range(modelClass.p['rows'] * modelClass.p['cols']):
        plot(modelClass.PC.x[i] / meter,modelClass.PC.y[i] / meter, color + '.', alpha= my_list[i])

    xlim(-10, modelClass.p['rows'])
    ylim(0, modelClass.p['cols'])
    xlabel('x')
    ylabel('y', rotation='vertical')
    axis('equal')
    title("Distance plot")
    show()


###### Cells activity ######

def reshape_spiking_times_(my_spikemon: SpikeMonitor,  my_spiking_index : int = 5, lower_threshold: float =0, upper_threshold=np.infty) -> Tuple[list, int, int]:
    """
    Given a SpikeMonitor object, returns a list of floats and nans, that represent the spike timing of the cells
    that are nan  if the timing is not inside [lower_threshold,upper_threshold ]. Here does not matter which index it is.
    :param my_spikemon: A spikemonitor object
    :type my_spikemon: SpikeMonitor
    :param lower_threshold: lower_threshold : the list will return spike timing that are greater than lower_threshold
    :type lower_threshold: float
    :param upper_threshold: upper_threshold : the list will return spike timing that are lower than upper_threshold
    :type upper_threshold: float or np.infty
    :return: [List of floats and nans that represent the spiking time inside bounds, argmin of the spikes,
    argmax of the spikes]
    :rtype:Tuple[list, int, int]
    """
    result = []
    min_ = inf
    max_ = -1.

    argmin_ = 0
    argmax_ = 0

    n = len(my_spikemon.values('t'))
    i = 0
    for i in range(n):
        my_result = nan
        for spiking_index in range(my_spiking_index):
            if len(my_spikemon.values('t')[i] / ms) > spiking_index:
                spike_time = my_spikemon.values('t')[i][spiking_index] / ms

                if spike_time >= lower_threshold and spike_time <= upper_threshold:
                    my_result = spike_time
                    if spike_time <= min_:
                        argmin_ = i
                        min_ = spike_time
                    elif spike_time >= max_:
                        argmax_ = i
                        max_ = spike_time
                else:
                    my_result = nan
        result.append(my_result)

    if len(result) == 0:
        raise Warning("No spikes")

    return result, argmin_, argmax_

def reshape_spiking_times(my_spikemon:SpikeMonitor, spiking_index:int=0, lower_threshold:float=0, upper_threshold=np.infty) -> Tuple[list, int, int]:
    """
    Given a SpikeMonitor object, returns a list of floats and nans, that represent the spike timing of the cells.
    When it's their [spiking_index]'s spike. Meaning if spiking_index=0, then we look only at the spikes that were the first ones.
    In the returned list, elements are nan  if the timing is not inside [lower_threshold,upper_threshold ].
    :param my_spikemon: A spikemonitor object
    :type my_spikemon: SpikeMonitor
    :param spiking_index: index representing which spike we're looking at.
    :type spiking_index: int
    :param lower_threshold: lower_threshold : the list will return spike timing that are greater than lower_threshold
    :type lower_threshold: float
    :param upper_threshold: upper_threshold : the list will return spike timing that are lower than upper_threshold
    :type upper_threshold: float or np.infty
    :return: [List of floats and nans that represent the spiking time inside bounds, argmin of the spikes,
    argmax of the spikes]
    :rtype:Tuple[list, int, int]
    """
    result = []
    min_ = inf
    max_ = -1.

    argmin_ = 0
    argmax_ = 0

    n = len(my_spikemon.values('t'))
    for i in range(n):
        if len(my_spikemon.values('t')[i] / ms) <= spiking_index:
            result.append(nan)
        else:
            spike_time = my_spikemon.values('t')[i][spiking_index] / ms
            if spike_time >= lower_threshold and spike_time <= upper_threshold:
                result.append(spike_time)
                if spike_time <= min_:
                    argmin_ = i
                    min_ = spike_time
                elif spike_time >= max_:
                    argmax_ = i
                    max_ = spike_time
            else:
                result.append(nan)

    if len(result) == 0:
        raise Warning("No spikes")

    return result, argmin_, argmax_




def test_spike_times(modelClass,  filePathName : str ="./video_spikes.mp4") -> Tuple[list, int, int]:
    """
    Creates a video of the spiking events, with the function: reshape_spiking_times_.
    Meaning it does not choose an index for the spiking time, but just gives it in firing order.
    :param modelClass: Network model
    :type modelClass: Either ThresholdModel or FairhallModel
    :param filePathName: path where to upload the video
    :type filePathName: str
    :return: [list of frames, height, width]
    :rtype: Tuple[list, int, int]
    """
    if not modelClass.has_run:
        raise ValueError("No spiking thus cannot compute the spike times")

    n = modelClass.p['rows'] * modelClass.p['cols'] - 1

    height = np.int(modelClass.PC.x[n] / meter)
    width = np.int(modelClass.PC.y[n] / meter)


    list_frames = []
    # TODO: duration

    duration = 100
    for j in range(duration):
        # Create the frame
        my_spikemon, argmin_, argmax_ = reshape_spiking_times_(modelClass.spikemon,lower_threshold=(j-1) , upper_threshold= (j + 1) )

        list_matrix = normalize(my_spikemon, 10, 255)
        if len(list_matrix[0]) > 0:
            for i in range(len(list_matrix[0])):
                if np.isnan(list_matrix[0][i]):
                    list_matrix[0][i] = 0
                else:
                    list_matrix[0][i] = 255

            Excitatory_matrix = np.zeros([np.int(modelClass.PC.y[n] / meter), np.int(modelClass.PC.x[n] / meter)])

            for i in range(modelClass.p['rows'] * modelClass.p['cols']):
                variable = list_matrix[0][i]
                Excitatory_matrix[np.int(modelClass.PC.y[n] / meter) - np.int(modelClass.PC.y[i] / meter) - 1, np.int(modelClass.PC.x[i] / meter) - 1] = variable
            list_frames.append(Excitatory_matrix)

    convert_to_movie(list_frames, height=height, width=width, filePathName=filePathName)

    return list_frames, height, width



def video_spike_times(modelClass, spiking_index:int =0, filePathName:str ="./video_spikes.mp4"):
    """
    Creates a video of the spiking events, with the function: reshape_spiking_times.
    The spikes will be showed in order of they spiked: First we show all the first spikes, then all the second spikes, etc..
    This indices of spikes will be bounded by spiking_index.
    :param modelClass: Network model
    :type modelClass: Either ThresholdModel or FairhallModel
    :param spiking_index: Bounds the indices of spikes shown in the video
    :type spiking_index: int
    :param filePathName: path where to upload the video
    :type filePathName: str
    :return: [list of frames, height, width]
    :rtype: Tuple[list, int, int]
    """
    if not modelClass.has_run:
        raise ValueError("No spiking thus cannot compute the spike times")

    n = modelClass.p['rows'] * modelClass.p['cols'] - 1

    height = np.int(modelClass.PC.x[n] / meter)
    width = np.int(modelClass.PC.y[n] / meter)


    list_frames = []
    for index_ in range(spiking_index + 1):
        for j in range(100):
            # Create the frame
            my_spikemon, argmin_, argmax_ = reshape_spiking_times(modelClass.spikemon, spiking_index=index_ ,lower_threshold=(j-1) , upper_threshold= (j + 1) )

            list_matrix = normalize(my_spikemon, 10, 255)
            if len(list_matrix[0]) > 0:
                for i in range(len(list_matrix[0])):
                    if np.isnan(list_matrix[0][i]):
                        list_matrix[0][i] = 0
                    else:
                        list_matrix[0][i] = 250

                Excitatory_matrix = np.zeros([np.int(modelClass.PC.y[n] / meter), np.int(modelClass.PC.x[n] / meter)])

                for i in range(modelClass.p['rows'] * modelClass.p['cols']):
                    variable = list_matrix[0][i]
                    Excitatory_matrix[np.int(modelClass.PC.y[n] / meter) - np.int(modelClass.PC.y[i] / meter) - 1, np.int(modelClass.PC.x[i] / meter) - 1] = variable
                list_frames.append(Excitatory_matrix)

    convert_to_movie(list_frames,  height=height,width= width, filePathName=filePathName)

    return list_frames, height, width


def create_movie(modelClass,spiking_index : Union[int, None]= 0, file_path_name: str = "video_spikes.mp4" ):
    """
    Creates a Movie of the spiking times and shows it.
    :param modelClass: Network model
    :type modelClass: Either ThresholdModel or FairhallModel
    :param spiking_index: Index of spikes shown in the video
    :type spiking_index: int
    :param file_path_name: file path name for the image that will create the connections
    :type file_path_name: str
    """
    if spiking_index == None:
        list_frames, height, width = test_spike_times(modelClass)
    else:
        list_frames, height, width = video_spike_times(modelClass, spiking_index=spiking_index)

    convert_to_movie(list_frames, height, width, file_path_name)
    width_ = 30 * modelClass.p['rows']
    height_ = 30 * modelClass.p['cols']
    Video.reload(file_path_name)
    Video(file_path_name, width=width_, height=height_)
    return file_path_name, width_, height_


def plot_spike_times(modelClass, spiking_index : int =0, threshold : float =0, plot_distribution : bool =True):
    """
    Plots the spiking events, with a gradient color indicating the spiking times association with the spiking_index.
    With the function: reshape_spiking_times.
    If spiking_index=0, it shows all the first spikes, if spiking_index=1, it shows all the second spikes
    This indices of spikes is given by spiking_index.
    :param modelClass: Network model
    :type modelClass: Either ThresholdModel or FairhallModel
    :param spiking_index: which index to plot
    :type spiking_index: int
    :param threshold: Lower threshold: show only the spikes greated than this threshold
    :type threshold: float
    :param plot_distribution: If histogram plot the distribution of spikes
    :type plot_distribution: bool
    """
    # If no cell spikes, cannot plot the spiking times.
    # But for later plotting reasons, still need to return indices of first and last spiking times.
    # Thus returns (arbitrarily 0 and 1).
    if not modelClass.has_run:
        raise ValueError("Class hasn't run thus cannot compute the spike times")

    color = 'r'

    my_spikemon, argmin_, argmax_  = reshape_spiking_times(modelClass.spikemon, spiking_index=spiking_index, lower_threshold=threshold)
    modelClass.reshaped_spikemon = my_spikemon
    normalized_times, min_, max_ = normalize(my_spikemon, 0, 0.9)


    if len(normalized_times) >0:

        if plot_distribution:
            plot_distrib(my_spikemon, "spiking times", my_xlabel="Spiking time in ms")
        # Just to plot the entire rectangle
        n = modelClass.p['rows'] * modelClass.p['cols'] - 1
        plot(modelClass.PC.x[0] / meter, modelClass.PC.y[0] / meter, '.', color='w')
        plot(modelClass.PC.x[n] / meter, modelClass.PC.y[n] / meter, '.', color='w')


        for j in range(modelClass.p['rows'] * modelClass.p['cols']):
            plot(modelClass.PC.x[j] / meter , modelClass.PC.y[j] / meter, color + '.', alpha = 1 - normalized_times[j])


        neuron_idx = modelClass.my_indices[0]
        plot(modelClass.PC.x[neuron_idx] / meter, modelClass.PC.y[neuron_idx] / meter, '*',
                  label=str(neuron_idx))

        # Marks first and last to spike (among the ones that actually spiked).
        plot(modelClass.PC.x[argmin_] / meter, modelClass.PC.y[argmin_] / meter, 'x', color='m', label="first")
        plot(modelClass.PC.x[argmax_] / meter, modelClass.PC.y[argmax_] / meter, 'x', color='k', label="last")

        xlim(-10, modelClass.p['rows'])
        ylim(0, modelClass.p['cols'])
        xlabel('x in meter')
        ylabel('y in meters', rotation='vertical')
        axis('equal')
        title("Spiking time of cells")
        legend()
        show()
    else:
        print("No neuron spiked.")

    # Updates the features of the spikes
    modelClass.number_spiking_cells, modelClass.mean_spike_times, modelClass.variance_spike_times =\
        variance_mean_spike_times(modelClass)
    modelClass.number_spiking_outside_trajectory, modelClass.mean_spikes_outside_trajectory, \
        modelClass.variance_spikes_outside_trajectory = number_spikes_outside_trajectoy(modelClass)


def variance_mean_spike_times(modelClass) -> Tuple[int, float, float]:
    """
    Computes the variance, and mean and number of the spike_times and updates the attribute of the class
    :param modelClass: Network model
    :type modelClass: Either ThresholdModel or FairhallModel
    :return: Tuple[number of spikes, mean[spike times], variance[spike times]]
    :rtype: Tuple[int, float, float]
    """
    my_list = [i for i in modelClass.reshaped_spikemon if not np.isnan(i)]

    modelClass.number_spiking_cells = len(my_list)
    modelClass.mean_spike_times = np.mean(my_list)
    modelClass.variance_spike_times = np.var(my_list)
    return len(my_list), np.mean(my_list), np.var(my_list)


def number_spikes_outside_trajectoy(modelClass)-> Tuple[int, float, float]:
    """
    Computes the variance, and mean and number of the spike_times outside the trajectory and updates the attribute of the class
    :param modelClass: Network model
    :type modelClass: Either ThresholdModel or FairhallModel
    :return: Tuple[number of spikes, mean[spike times], variance[spike times]]
    :rtype:Tuple[int, float, float]
    """
    color = 'b'
    result = []
    for i in range(modelClass.p['rows'] * modelClass.p['cols']):
        if (not np.isnan(modelClass.reshaped_spikemon[i]) ) and modelClass.p['trajectory'][0,i] == 0 :
            plot(modelClass.PC.x[i] / meter, modelClass.PC.y[i] / meter, color + '.')
            result.append(modelClass.reshaped_spikemon[i])

    title('Cells that spike outside the trajectory')
    xlabel('m')
    ylabel('m', rotation='vertical')
    show()
    modelClass.number_spiking_outside_trajectory = len(result)
    modelClass.mean_spikes_outside_trajectory = np.mean(result)
    modelClass.variance_spikes_outside_trajectory = np.var(result)
    return len(result), np.mean(result), np.var(result)

# Computes first spiking times of neurons, given their voltages recording.

def spiking_times_fun(modelClass, type_ : str ='PC'):
    """
    Updates first and last spike of a cell of type "type_".
    :param modelClass: Network model
    :type modelClass: Either ThresholdModel or FairhallModel
    :param type_: type of cell
    :type type_: str
    """
    if not modelClass.has_run:
        raise ValueError("Cannot record the spiking times if the network did not ran yet.")


    if type_ == 'PC':
        modelClass.PC_first_spike = reshape_spiking_times(modelClass.spikemon)[1]
        modelClass.PC_last_spike = reshape_spiking_times(modelClass.spikemon)[2]
    elif type_ == 'G':
        modelClass.G_first_spike = reshape_spiking_times(modelClass.spikemong)[1]
        modelClass.G_last_spike = reshape_spiking_times(modelClass.spikemong)[2]
    elif type_ == 'S':
        modelClass.S_first_spike = reshape_spiking_times(modelClass.spikemons)[1]
        modelClass.S_last_spike = reshape_spiking_times(modelClass.spikemons)[2]
    elif type_ == 'INH':
        modelClass.INH_first_spike =reshape_spiking_times(modelClass.spikemoninh)[1]
        modelClass.INH_last_spike = reshape_spiking_times(modelClass.spikemoninh)[2]

def plot_voltages_PC(modelClass, plot_last_first : bool =True, new_indices : list =[]):
    """
    Plot the voltage of some Pyramidal cells.
    :param modelClass: Network model
    :type modelClass: Either ThresholdModel or FairhallModel
    :param plot_last_first: If plot the last and first spike.
    :type plot_last_first: bool
    :param new_indices: list of indices of cells for which we want to see the voltage.
    :type new_indices: list
    """
    # Checks it has indeed ran
    if not modelClass.has_run:
        raise ValueError("Cannot plot if hasn't plot")

    if len(new_indices)>0:
        my_indices = new_indices
    else:
        my_indices = modelClass.my_indices

    for i in range(len(my_indices)):
        my_plot = plot(modelClass.MPC.t / ms, modelClass.MPC.v[i], label='PC' + str(my_indices[i]))
        index = my_indices[i]
        if index in list(modelClass.spikemon.i):
            plot(modelClass.PC_all_values['t'][index] / ms, modelClass.PC_all_values['v'][index], 'o', color=my_plot[0].get_color() )

    if plot_last_first:
        first_plot = plot(modelClass.MM.t / ms, modelClass.MM.v[modelClass.PC_first_spike], label='First Spike' + str(modelClass.PC_first_spike))

        plot(modelClass.PC_all_values['t'][modelClass.PC_first_spike] / ms, modelClass.PC_all_values['v'][modelClass.PC_first_spike], 'o',
             color=first_plot[0].get_color())
        last_plot = plot(modelClass.MM.t / ms, modelClass.MM.v[modelClass.PC_last_spike], label='Last Spike' + str(modelClass.PC_last_spike))

        plot(modelClass.PC_all_values['t'][modelClass.PC_first_spike] / ms, modelClass.PC_all_values['v'][modelClass.PC_first_spike], 'o',
             color=last_plot[0].get_color())
    legend()
    title("Voltage of Pyramidal cells")
    xlabel("Time in ms")
    ylabel("Voltage", rotation='vertical')
    show()

def plot_voltages_other_types(modelClass, type_list=['R','N', 'INH'], my_indices=[0], plot_last_first=True):
    """
    Plot the voltage of some cells of type given in the list type_list.
    :param modelClass: Network model
    :type modelClass: Either ThresholdModel or FairhallModel
    :param type_list: list of the type we want to plot
    :type type_list: list[str]
    :param new_indices: list of indices of cells for which we want to see the voltage.
    :type new_indices: list
    """

    if not modelClass.has_run:
        raise ValueError("Cannot plot if hasn't ran")

    my_title = 'Voltage of '

    for type_ in type_list:
        my_title = my_title + type_ + ' '

        if type_ == 'threshold':
            for i in my_indices:
                my_plot = plot(modelClass.Mthreshold.t / ms, modelClass.Mthreshold.h[i],
                               label='Threshold for PC ' + str(i) + 'cell.')

            # plot(modelClass.MthresholdG.t / ms, modelClass.MthresholdG.h[0], label='Threshold for G cell.')
        elif type_ == 'weights':
            plot(modelClass.weights[0].t / ms, modelClass.weights[0].w, label='Weight[' + str(0) + ']')
            if modelClass.model == "FairhallModel":
                plot(modelClass.weights[0].t / ms, modelClass.weights[0].w, label='Weight in the trajectory')

                plot(modelClass.weights2[0].t / ms, modelClass.weights2[0].w, label='Weight outside the trajectory')


        elif type_ == 'INH':
                for i in my_indices:
                    my_plot = plot(modelClass.MINH.t / ms, modelClass.MINH.v[i], label='INH' + str(i))
                plot(modelClass.INH_all_values['t'][0] / ms, modelClass.INH_all_values['v'][0], 'o', color=my_plot[0].get_color())

        elif type_ == 'N':
            for i in my_indices:
                my_plot = plot(modelClass.MN.t / ms, modelClass.MN.v[i], label='N' + str(i))
            plot(modelClass.N_all_values['t'][0] / ms, modelClass.N_all_values['v'][0], 'o', color=my_plot[0].get_color())
        elif type_ == 'R':
            for i in my_indices:
                my_plot = plot(modelClass.MR.t / ms, modelClass.MR.v[i], label='R' + str(i))
            plot(modelClass.R_all_values['t'][0] / ms, modelClass.R_all_values['v'][0], 'o',
                 color=my_plot[0].get_color())
        elif type_ == 'PC':
            plot_voltages_PC(modelClass,new_indices=my_indices, plot_last_first=plot_last_first)

        else:
            raise ValueError("Type must be 'PC', 'R' , 'INH', 'threshold' or 'N' ")

    title(my_title)
    xlabel("Time in ms")
    ylabel("Voltage", rotation='vertical')
    if len(my_indices) <= 5 :
        legend()
    show()

def add_params(params, rec_weight=4.5, ext_weight=0.5, R_weight=0.4, inh_weight_pi=0.02, inh_weight_ip=0.01):
    """
    Copy the parameters, and return a modify dictionary of parameters with new ones.
    :param params: parameters
    :type params: dict
    :param rest : Different parameters
    :type rest: float
    :return:
    :rtype:
    """
    new_params = copy.deepcopy(params)
    new_params['rec_weight'] = rec_weight
    new_params['ext_weight'] = ext_weight
    new_params['R_weight'] = R_weight
    new_params['inh_weight_pi'] = inh_weight_pi
    new_params['inh_weight_ip'] = inh_weight_ip
    return new_params

def create_trajectory_matrix(file_path_name:str = 'S_inputs.npy', num_ext_neurons:int = 1) -> np.array:
    """
    Creates an S matrix from a given image (located with the file_path_name)
    # TODO : Check for the number of neurons and get it better.
    :param num_ext_neurons: number of S neurons (obsolete)
    :type num_ext_neurons: int
    :param file_path_name: file path name for the image that will create the connections
    :type file_path_name: str
    :return: Matrix with column[i] is 1 if i is in the trajectory (if was painted), 0 otherwise
    :rtype: np.array
    """
    trajectory = np.load(file_path_name)
    trajectory = 255 * np.ones(trajectory.shape) - trajectory
    result = convert_matrix_pos_to_indices(num_ext_neurons, trajectory, rows=20, cols=30)
    return result

def plot_run(params, model="FairhallModel", plasticity=True, my_duration=100, record_=True, plot_last_first=False):
    """
    Runs the network and plots some 1- Pyramidal cells activity 2- The spike times with gradient color
    :param params: parameters necessary to the construction of the model
    :type params: dict
    :param modelClass: Network Model
    :type modelClass: Either ThresholdModel or Fairhall Model
    :param my_duration: duration of the simulation
    :type my_duration: int
    :return: The model
    :rtype: Either ThresholdModel or Fairhall Model
    """
    start_scope()

    fm1 = Model(params, model=model, plasticity=plasticity)

    fm1.run(duration=my_duration * ms, show_PC=True, show_other=False, record_=record_, plot_last_first=plot_last_first)

    plot_spike_times(fm1, 0, plot_distribution=False)


    return fm1


#def visualize_hyperparameters(Matrix_2D:np.array):

def visualize_hyperparameters(Matrix_2D:np.array,my_extent, title_:str):
    fig, (ax1) = plt.subplots(figsize=(3,3), ncols=1)
    pos = ax1.imshow(Matrix_2D, cmap='Blues',
     extent=my_extent, interpolation='none')
    cbar = fig.colorbar(pos, ax=ax1)
    cbar.set_label('Label name',size=3)
    cbar.minorticks_on()
    plt.xlabel('meters')
    plt.ylabel('meters', rotation="vertical")
    plt.title(title_)
    plt.show()


def explore_hyperparameters(my_dict:dict, step:list, model:str, trajectory, n:int = 2):
    if len(list(my_dict.keys())) != 2:
        if len(step) != 2:
            raise Warning("Dictionary and step must be the same size!")

    list_i = []
    for i in range(n):
        list_j = []
        for j in range(n):
            p = {}
            p[list(my_dict.keys())[0]] = list(my_dict.values())[0] + i * step[0]
            p[list(my_dict.keys())[1]] = list(my_dict.values())[1] + j * step[1]
            p['trajectory'] = trajectory
            fm1 = plot_run(p, model, record_=False)
            list_j.append(fm1)
        list_i.append(list_j)

    num_spiking_cells = np.zeros([n, n])
    mean_spike = np.zeros([n, n])
    var_spike_times = np.zeros([n, n])
    num_spiking_outside_trajectory = np.zeros([n, n])
    mean_spikes_outside_traj = np.zeros([n, n])
    var_spikes_outside_trajectory = np.zeros([n, n])

    for i in range(n):
        for j in range(n):
            num_spiking_cells[i, j] = list_i[i][j].number_spiking_cells
            mean_spike[i, j] = list_i[i][j].mean_spike_times
            var_spike_times[i, j] = list_i[i][j].variance_spike_times
            num_spiking_outside_trajectory[i, j] = list_i[i][j].number_spiking_outside_trajectory
            mean_spikes_outside_traj[i, j] = list_i[i][j].mean_spikes_outside_trajectory
            var_spikes_outside_trajectory[i, j] = list_i[i][j].variance_spikes_outside_trajectory

    my_extent = [list(my_dict.values())[0], list(my_dict.values())[0] + (n-1) * step[0], list(my_dict.values())[1] + (n-1) * step[1], list(my_dict.values())[1]]
    visualize_hyperparameters(num_spiking_cells, my_extent, "num_spiking_cells")

    visualize_hyperparameters(mean_spike, my_extent, "mean_spike")

    visualize_hyperparameters(var_spike_times, my_extent, "var_spike_times")

    visualize_hyperparameters(num_spiking_outside_trajectory, my_extent, "num_spiking_outside_trajectory")

    visualize_hyperparameters(mean_spikes_outside_traj, my_extent, "mean_spikes_outside_traj")

    visualize_hyperparameters(var_spikes_outside_trajectory, my_extent, "var_spikes_outside_trajectory")

    return num_spiking_cells, mean_spike, var_spike_times, num_spiking_outside_trajectory, mean_spikes_outside_traj, var_spikes_outside_trajectory




def time_me(func):
    def wrapper(*args, **kwargs):
        t = time()
        print("Started", func.__name__)
        result = func(*args, **kwargs)
        print("Finished", func.__name__, "in", time() - t)
        return result
    return wrapper


def test_(spikemon, modelClass, filePathName:str = "./video_spikes.mp4", simTime=100, nBins=5):

    if not modelClass.has_run:
        raise ValueError("No spiking thus cannot compute the spike times")

    n = modelClass.p['rows'] * modelClass.p['cols'] - 1


    height = np.int(modelClass.PC.x[n] / meter)
    width = np.int(modelClass.PC.y[n] / meter)

    list_frames = []
    nX = modelClass.p['rows']
    nY = modelClass.p['cols']

    resArr, t = spike_times_to_matrix(modelClass, spikemon, nX, nY, 0 * ms, simTime * ms, nBins)

    return resArr, t

@time_me
def spike_times_to_matrix(modelClass, spikeMon, nX, nY, startTime, endTime, nBins):

    n = nX * nY -1
    height = np.int(modelClass.PC.x[n] / meter)
    width = np.int(modelClass.PC.y[n] / meter)


    resArr = np.zeros((width, height, nBins))
    tStep = float((endTime - startTime) / nBins)

    coord1Dto2D = lambda i: (np.int(modelClass.PC.y[n] / meter) - np.int(modelClass.PC.y[i] / meter) - 1, np.int(modelClass.PC.x[i] / meter) - 1)
    time2bin = lambda t: int((t - startTime) / tStep)

    for iNeuron, tList in spikeMon.all_values()['t'].items():
        x, y = coord1Dto2D(iNeuron)

        for t in tList:
            my_bin = time2bin(t)
            resArr[x, y, my_bin] += 1

    return resArr

def new_spike_times(modelClass, filePathName:str = "./video_spikes.mp4", simTime=None, nBins=None):
    """
    Creates a video of the spiking events, with the function: reshape_spiking_times_.
    Meaning it does not choose an index for the spiking time, but just gives it in firing order.
    :param modelClass: Network model
    :type modelClass: Either ThresholdModel or FairhallModel
    :param filePathName: path where to upload the video
    :type filePathName: str
    :return: [list of frames, height, width]
    :rtype: Tuple[list, int, int]
    """


    if not modelClass.has_run:
        raise ValueError("No spiking thus cannot compute the spike times")

    if simTime == None:
        simTime = modelClass.duration / ms
    print(simTime)
    if nBins == None:
        nBins = int(simTime)
    print(nBins)

    n = modelClass.p['rows'] * modelClass.p['cols'] - 1



    height = np.int(modelClass.PC.x[n] / meter)
    width = np.int(modelClass.PC.y[n] / meter)

    list_frames = []
    nX = modelClass.p['rows']
    nY = modelClass.p['cols']

    arr = 255 * spike_times_to_matrix(modelClass, modelClass.spikemon, nX , nY, 0 * ms, simTime * ms, nBins)

    for i in range(arr.shape[2]):
        list_frames.append(arr[:, :, i])

    base_matrix = convert_to_movie(list_frames, height=height , width=width, filePathName=filePathName)

    return base_matrix, height, width


def create_movie_(modelClass,spiking_index : Union[int, None]= 0, file_path_name: str = "video_spikes.mp4" ):
    """
    Creates a Movie of the spiking times and shows it.
    :param modelClass: Network model
    :type modelClass: Either ThresholdModel or FairhallModel
    :param spiking_index: Index of spikes shown in the video
    :type spiking_index: int
    :param file_path_name: file path name for the image that will create the connections
    :type file_path_name: str
    """
    if spiking_index == None:
        list_frames, height, width = test_spike_times(modelClass)
    else:
        list_frames, height, width = video_spike_times(modelClass, spiking_index=spiking_index)

    convert_to_movie(list_frames, height, width, file_path_name)
    width_ = 30 * modelClass.p['rows']
    height_ = 30 * modelClass.p['cols']
    Video.reload(file_path_name)
    Video(file_path_name, width=width_, height=height_)
    return file_path_name, width_, height_
