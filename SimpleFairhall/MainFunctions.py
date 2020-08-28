from brian2 import ms, start_scope
from Model import Model
import numpy as np
import copy
import ConvertingFunctions, CellActivityFunctions

############## Initialisation #####################

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
    result = ConvertingFunctions.convert_matrix_pos_to_indices(num_ext_neurons, trajectory, rows=20, cols=30)
    return result

def plot_run(params, model="FairhallModel", plasticity=True, my_duration=100, record_=True):
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

    fm1.run(duration=my_duration * ms, show_PC=True, show_other=False, record_=record_)

    CellActivityFunctions.plot_spike_times(fm1, 0, plot_distribution=False)


    return fm1
