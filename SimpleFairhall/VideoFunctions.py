from time import time
import numpy as np
from brian2 import*
import ConvertingFunctions
########### Video #################


def time_me(func):
    def wrapper(*args, **kwargs):
        t = time()
        print("Started", func.__name__)
        result = func(*args, **kwargs)
        print("Finished", func.__name__, "in", time() - t)
        return result
    return wrapper

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

def video_spike_times(modelClass, filePathName:str = "./video_spikes.mp4", simTime=None, nBins=None):
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

    base_matrix = ConvertingFunctions.convert_to_movie(list_frames, height=height , width=width, filePathName=filePathName)

    return base_matrix, height, width


def new_spike_times(modelClass, filePathName: str = "./video_spikes.mp4", simTime=None, nBins=None):
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

    arr = 255 * spike_times_to_matrix(modelClass, modelClass.spikemon, nX, nY, 0 * ms, simTime * ms, nBins)

    list_frames.append(arr[:,:,0])
    list_frames.append(arr[:, :, 1] + 0.5 * arr[:, :,0])
    list_frames.append(arr[:, :, 2] + 0.5 * arr[:, :, 1] + 0.25 * arr[:, :, 0])
    for i in range(3,arr.shape[2]):
        list_frames.append(arr[:, :, i] + 0.5 * arr[:,:,i-1] + 0.25 * arr[:,:,i-2] + 0.125 * arr[:,:,i-3])

    list_frames = ConvertingFunctions.convert_to_movie(list_frames, height=height, width=width,
                                                       filePathName=filePathName)

    return list_frames, height, width

