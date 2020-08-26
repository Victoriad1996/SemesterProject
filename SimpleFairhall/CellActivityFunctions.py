from brian2 import*
import BasicFunctions
import numpy as np
from typing import Tuple


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
    normalized_times, min_, max_ = BasicFunctions.normalize(my_spikemon, 0, 0.9)


    if len(normalized_times) >0:

        if plot_distribution:
            BasicFunctions.plot_distrib(my_spikemon, "spiking times", my_xlabel="Spiking time in ms")
        # Just to plot the entire rectangle
        n = modelClass.p['rows'] * modelClass.p['cols'] - 1
        plot(modelClass.PC.x[0] / meter, modelClass.PC.y[0] / meter, '.', color='w')
        plot(modelClass.PC.x[n] / meter, modelClass.PC.y[n] / meter, '.', color='w')


        for j in range(modelClass.p['rows'] * modelClass.p['cols']):
            plot(modelClass.PC.x[j] / meter , modelClass.PC.y[j] / meter, color + '.', alpha = 1 - normalized_times[j])


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
