from brian2 import exp, meter
from matplotlib.pyplot import*
from numpy import*

# Returns a list of indices. Will be useful for plotting.
def some_indices(neuron_idx=50):
    neuron_idx2 = neuron_idx - 1
    neuron_idx3 = neuron_idx + 1
    neuron_idx4 = neuron_idx + 20
    other_idx = 200
    my_indicies = [neuron_idx, neuron_idx2, neuron_idx3, neuron_idx4, other_idx]
    return my_indicies

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

# Plots the distribution of any list.
def plot_distrib(my_list, name_title):
    hist(my_list, color = 'blue', edgecolor = 'black',
         bins = int(180/5))
    title("Distribution of " + str(name_title))

# Computes the distance of pyramidal neurons of indices : neuron_idx and j.
def distance(PC, neuron_idx, j):
    #result = exp(-((PC.x[neuron_idx] - PC.x[j]) ** 4 + (PC.y[neuron_idx] - PC.y[j]) ** 4) / (2*15*meter)**4)
    #result = ((PC.x[neuron_idx] - PC.x[j]) ** 2 + (PC.y[neuron_idx] - PC.y[j]) ** 2) / (2*15*meter)**2
    result = (np.abs(PC.x[neuron_idx] - PC.x[j]) + np.abs(PC.y[neuron_idx] - PC.y[j])) / (meter)

    return result

# Computes the distance of all pyramidal cells to the neuron_idx. Uses the function distance(PC, neuron_idx, j).
def list_distance(PC, neuron_idx):
    n = len(PC.x)
    list_dist = [distance(PC, neuron_idx, j) for j in range(n)]
    max_ = max(list_dist)
    result = [distance(PC, neuron_idx, i)/max_ for i in range(n)]
    return result

# If gives a neuron_idx, then plots all the pyramidal cells it's connected to by the synapses S.
# Otherwise plots all neurons in the synapses S.
def plot_connectivity(PC, S, rows, cols, my_title, neuron_idx = None):
    color = 'b'
    # Plots all the pyramidal cells it's connected to by the synapses S.
    if neuron_idx == None:
        for i in range(rows*cols):
            plot(PC.x[i] / meter,PC.y[i] / meter, color + '.', alpha = 0.8)

        plot(S.x / meter, S.y / meter, 'r' + '.', label="Connected neurons")

    # Plots all neurons in the synapses S.
    else:
        # The neurons connected by synapses S to neuron_idx
        plot(S.x[neuron_idx,:] / meter, S.y[neuron_idx,:] / meter, 'r' + '.', label="Connected neurons")
        # Plots the neuron_idx.
        plot(PC.x[neuron_idx] / meter, PC.y[neuron_idx] / meter, color + '.', alpha=0.8, label="Main neuron")

    legend()
    title("Connectivity between PC and " + str(my_title))

# Plot, with gradient color, the distance of each neurons to the "neuron_idx" neuron given in parameters.
def plot_distance(PC, neuron_idx, rows, cols):

    # Computes the distance of all PC cells to the neuron_idx and returns a list.
    my_list = list_distance(PC, neuron_idx)
    # Plots the distribution of this list of distances.
    plot_distrib(my_list, "distances")
    show()

    # Marks the neuron_idx
    plot(PC.x[neuron_idx] / meter, PC.y[neuron_idx] / meter, 'o',
                mfc='none', label=str(neuron_idx))
    color = 'b'

    # The alpha allows to "grade" the distance to the neuron cell.
    for i in range(rows*cols):
        plot(PC.x[i] / meter,PC.y[i] / meter, color + '.', alpha= my_list[i])

    xlim(-10, rows)
    ylim(0, cols)
    xlabel('x')
    ylabel('y', rotation='horizontal')
    axis('equal')

    title("Distance plot")
    show()


#Plots the first spiking times of each cell.
def plot_spike_times(modelClass):
    print("model spiking times ", modelClass.spiking_times)

    color = 'r'
    # Normalises the spiking times such that. Then it will be used for the "alpha" in plotting.
    # argmin_, and argmax_ returned so we can mark those cells in the plot.
    norm_l, argmin_, argmax_ = normalize(modelClass.spiking_times)

    # Plot the distribution ot the spiking times. Useful to see if the plot will be "readable": they are enough shades to understand the spiking order.
    plot_distrib(norm_l, "normalized spiking times")
    plot_distrib(modelClass.spiking_times, "spiking times")
    show()

    # Plots all points, the spiking time is "encoded" in alpha: more red it is, sooner it spikes.
    for i in range(modelClass.rows * modelClass.cols):
        if modelClass.spiking_times[i]<inf:
            plot(modelClass.PC.x[i] / meter, modelClass.PC.y[i] / meter, color + '.', alpha=1-norm_l[i])

    # Marks some indices (useful to mark them when looking at the plot of voltages).
    for i in range(1,len(modelClass.my_indices)):
        plot(modelClass.PC.x[modelClass.my_indices[i]] / meter, modelClass.PC.y[modelClass.my_indices[i]] / meter, 's', color='r',
             alpha=1- norm_l[modelClass.my_indices[i]], label=str(modelClass.my_indices[i]))

    # Marks the first index of the list.
    neuron_idx = modelClass.my_indices[0]
    plot(modelClass.PC.x[neuron_idx] / meter, modelClass.PC.y[neuron_idx] / meter, '*',
              label=str(neuron_idx))

    # Marks first and last to spike (among the ones that actually spiked).
    plot(modelClass.PC.x[argmin_] / meter, modelClass.PC.y[argmin_] / meter, 'x', color='k', label="first")
    plot(modelClass.PC.x[argmax_] / meter, modelClass.PC.y[argmax_] / meter, 'x', color='k', label="last")

    #color_gradient = plot(PC.x[new_indices] / meter, PC.y[new_indices] / meter, color + '.', alpha=float(alpha_indices)/m)

    xlim(-10, modelClass.rows)
    ylim(0, modelClass.cols)
    xlabel('x')
    ylabel('y', rotation='horizontal')
    axis('equal')
    title("Spiking time of cells")
    legend()
    show()

    # Returns argmin_, argmax_ for later plotting voltages of those neurons.
    return argmin_, argmax_

# Normalises a list of positive values to a list of values between 0 and 1.
# It helps for plotting spikes: for "alpha" needs values between 0 and 1.
def normalize(l):
    # Takes out the "inf" values in order to compute the max, min and length of all given values.
    norm_l = [i for i in l if i < inf]

    min_ = np.min(norm_l)
    argmin_ = np.argmin(norm_l)
    argmax_ = np.argmax(norm_l)
    length = np.max(norm_l) - min_
    result = []


    if length > 0:
        for i in l:

            if i < inf:
                result.append((i-min_)/length)
            # if they are some inf values, return them.
            else:
                result.append(i)

    # If all values are the same, then just returns 1s.
    else:
        for i in l:
            if i < inf:
                result.append(1)
            else:
                result.append(i)

    # Also returns argmin_, argmax_. It helps for knowing who spikes first and last (among the one that actually spike).
    return result, argmin_, argmax_



def sgmd(x):
    """Sigmoid (logistic) function."""

    return 1 / (1 + np.exp(-x))

# Computes first spiking times of neurons, given their voltages recording.
# TODO: Replace this by more efficient SpikeMonitor
def spiking_times_fun(MM, rows, cols, threshold):
    spiking_times = []

    for element in range(rows * cols):
        if len(list(np.where(MM.v[element] >= threshold))[0]) > 0:
            spike_time = list(np.where(MM.v[element] >= threshold))[0].tolist()[0]
            spiking_times.append(spike_time)
        else:
            # TODO: Check if it is better to append inf instead of nan.
            spiking_times.append(nan)


    return spiking_times

