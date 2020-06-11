from brian2 import exp, meter, ms
from matplotlib.pyplot import*
from numpy import*
from math import isnan
import warnings


######################### Basic Functions #########################

# Returns a list of indices. Will be useful for plotting.
def some_indices(neuron_idx=50):
    neuron_idx2 = neuron_idx - 1
    #neuron_idx3 = neuron_idx + 1
    neuron_idx3 = 16
    new_idx = 0
    rows = 20
    cols = 30

    for j in range(rows * cols):
        if (j//rows == 10 and j % rows == 5):
            new_idx = j

    my_indices = [neuron_idx, neuron_idx2, neuron_idx3, new_idx]
    return my_indices

# Computes the distance of pyramidal neurons of indices : neuron_idx and j.
def distance(PC, neuron_idx, j):
    #result = exp(-((PC.x[neuron_idx] - PC.x[j]) ** 4 + (PC.y[neuron_idx] - PC.y[j]) ** 4) / (2*15*meter)**4)
    result = ((PC.x[neuron_idx] - PC.x[j]) ** 2 + (PC.y[neuron_idx] - PC.y[j]) ** 2) / (meter)**2
    #
    #result = (np.abs(PC.x[neuron_idx] - PC.x[j]) + np.abs(PC.y[neuron_idx] - PC.y[j])) / (meter)

    return result

# Computes the distance of all pyramidal cells to the neuron_idx. Uses the function distance(PC, neuron_idx, j).
def list_distance(PC, neuron_idx):
    n = len(PC.x)
    list_dist = [distance(PC, neuron_idx, j) for j in range(n)]
    max_ = max(list_dist)
    result = [distance(PC, neuron_idx, i)/max_ for i in range(n)]
    return result

# Checks if a list is only nans
def is_list_nan(mylist):
    result = True
    for i in mylist:
        if not isnan(i):
            result = False
    return result

def is_list_inf(mylist):
    result = True
    for i in mylist:
        if not np.isinf(i):
            result = False
    return result

# Normalises a list of positive values to a list of values between 0 and 1.
# It helps for plotting spikes: for "alpha" needs values between 0 and 1.
def normalize(l, lower=0, upper=1):
    # Takes out the "inf" values in order to compute the max, min and length of all given values.
    if lower>=upper:
        raise ValueError("Lower key must be strictly smaller than upper key.")

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
def plot_distrib(my_list, name_title):
    hist(my_list, color = 'blue', edgecolor = 'black',
         bins = int(180/5))
    title("Distribution of " + str(name_title))
    show()

def sgmd(x):
    """Sigmoid (logistic) function."""

    return 1 / (1 + np.exp(-x))


######################### Plotting functions #########################


###### Network Structure ######

def plot_different_distances(modelClass):
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
def visualise_connectivity(S, mylegend):
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
    ylabel('Target neuron index')
    suptitle(mylegend)

# If gives a neuron_idx, then plots all the pyramidal cells it's connected to by the synapses S.
# Otherwise plots all neurons in the synapses S.
def plot_connectivity(modelClass, list_synapses , my_title, neuron_idx = None):
    color = 'b'
    # Plots all the pyramidal cells it's connected to by the synapses S.
    list_colors = ['r', 'k', 'g', 'y']
    j = 0
    if neuron_idx == None:
        for i in range(modelClass.p['rows']*modelClass.p['cols']):
            plot(modelClass.PC.x[i] / meter, modelClass.PC.y[i] / meter, color + '.', alpha = 0.8)
        for S in list_synapses:
            plot(S.x / meter, S.y / meter, list_colors[j] + '.', label="Connected neurons")
            j += 1

    # Plots all neurons in the synapses S.
    else:
        # The neurons connected by synapses S to neuron_idx
        for S in list_synapses:
            plot(S.x[neuron_idx,:] / meter, S.y[neuron_idx,:] / meter, 'r' + '.', label="Connected neurons")
        # Plots the neuron_idx.
        plot(modelClass.PC.x[neuron_idx] / meter, modelClass.PC.y[neuron_idx] / meter, color + '.', alpha=0.8, label="Main neuron")

    legend()
    title("Connectivity between PC and " + str(my_title))

# Plot, with gradient color, the distance of each neurons to the "neuron_idx" neuron given in parameters.
def plot_distance(modelClass):

    # Computes the distance of all PC cells to the neuron_idx and returns a list.
    my_list = list_distance(modelClass.PC, modelClass.neuron_idx)
    # Plots the distribution of this list of distances.
    plot_distrib(my_list, "distances")
    show()

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
    ylabel('y', rotation='horizontal')
    axis('equal')
    title("Distance plot")
    show()


###### Cells activity ######

def reshape_spiking_times (my_spikemon, spiking_index=0, threshold=0):
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
            if spike_time >= threshold:
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


def plot_spike_times(modelClass, spiking_index=0, threshold=0):
    # If no cell spikes, cannot plot the spiking times.
    # But for later plotting reasons, still need to return indices of first and last spiking times.
    # Thus returns (arbitrarily 0 and 1).
    if not modelClass.has_run:
        raise ValueError("No spiking thus cannot compute the spike times")

    color = 'r'

    my_spikemon, argmin_, argmax_  = reshape_spiking_times(modelClass.spikemon, spiking_index=spiking_index, threshold=threshold)
    normalized_times, min_, max_ = normalize(my_spikemon, 0, 0.9)
    plot_distrib(my_spikemon, "spiking times")
    show()

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
    xlabel('x')
    ylabel('y', rotation='horizontal')
    axis('equal')
    title("Spiking time of cells")
    legend()
    show()


# Computes first spiking times of neurons, given their voltages recording.

def spiking_times_fun(modelClass, type_='PC'):
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


def record_spikes(modelClass, threshold=-36.1, type_='PC', index=0):
    if not modelClass.has_run:
        raise ValueError("Cannot record the spiking times if the network did not ran yet.")

    spike_time = []
    if type_ == 'PC':
        if len(list(np.where(modelClass.MM.v[index] >= threshold))[0]) > 0:
            spike_time = list(np.where(modelClass.MM.v[index] >= threshold))[0].tolist()

        # Record spiking times of G inputs
    elif type_ == 'G':
        if len(list(np.where(modelClass.MG.v[0] >= threshold))) > 0:
            spike_time = list(np.where(modelClass.MG.v[0] >= threshold))[0].tolist()

        # Record spiking times of Inhibitory inputs
    elif type_ == 'INH':
        if len(list(np.where(modelClass.MINH.v[0] >= threshold))[0]) > 0:
            spike_time = list(np.where(modelClass.MINH.v[0] >= threshold))[0].tolist()

    return spike_time


def plot_voltages_PC(modelClass, plot_last_first=True, new_indices=[]):
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
        plot(modelClass.MM.t / ms, modelClass.MM.v[modelClass.PC_first_spike], label='First Spike' + str(modelClass.PC_first_spike))
        plot(modelClass.MM.t / ms, modelClass.MM.v[modelClass.PC_last_spike], label='Last Spike' + str(modelClass.PC_last_spike))

    legend()
    title("Voltage of Pyramidal cells")
    show()

def plot_voltages_other_types(modelClass, type_list=['G', 'S', 'INH'], my_indices=[0]):

    if not modelClass.has_run:
        raise ValueError("Cannot plot if hasn't ran")

    my_title = 'Voltage of '

    for type_ in type_list:
        my_title = my_title + type_ + ' '
        if type_ == 'INH':
            for i in my_indices:
                my_plot = plot(modelClass.MINH.t / ms, modelClass.MINH.v[i], label='INH' + str(i))
            plot(modelClass.INH_all_values['t'][0] / ms, modelClass.INH_all_values['v'][0], 'o', color=my_plot[0].get_color())
        elif type_ == 'G':
            for i in my_indices:
                my_plot = plot(modelClass.MG.t / ms, modelClass.MG.v[i], label='G' + str(i))
            plot(modelClass.G_all_values['t'][0] / ms, modelClass.G_all_values['v'][0], 'o', color=my_plot[0].get_color())

        elif type_ == 'S':
            for i in my_indices:
                my_plot = plot(modelClass.MS.t / ms, modelClass.MS.v[i], label='S' + str(i))
            plot(modelClass.S_all_values['t'][0] / ms, modelClass.S_all_values['v'][0], 'o',color=my_plot[0].get_color())
        else:
            raise ValueError("Type must be 'PC', 'G' , 'INH' or 'S' ")

    title(my_title)
    legend()
    show()
