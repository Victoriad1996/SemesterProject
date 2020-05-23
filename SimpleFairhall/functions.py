from brian2 import exp, meter
from matplotlib.pyplot import*
from numpy import*

def some_indices(neuron_idx):
    neuron_idx = 50
    neuron_idx2 = neuron_idx - 1
    neuron_idx3 = neuron_idx + 1
    neuron_idx4 = neuron_idx + 20
    other_idx = 200
    my_indicies = [neuron_idx, neuron_idx2, neuron_idx3, neuron_idx4, other_idx]
    return my_indicies
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

def plot_distrib(my_list, name_title):
    hist(my_list, color = 'blue', edgecolor = 'black',
         bins = int(180/5))
    title("Distribution of " + str(name_title))


def distance(PC, neuron_idx, j):
    #result = exp(((PC.x[neuron_idx] - PC.x[j]) ** 2 + (PC.y[neuron_idx] - PC.y[j]) ** 2) / (2*15*meter)**2) -1
    #result = ((PC.x[neuron_idx] - PC.x[j]) ** 2 + (PC.y[neuron_idx] - PC.y[j]) ** 2) / (2*15*meter)**2
    result = (np.abs(PC.x[neuron_idx] - PC.x[j]) + np.abs(PC.y[neuron_idx] - PC.y[j])) / (meter)

    return result

def list_distance(PC, neuron_idx):
    n = len(PC.x)
    list_dist = [distance(PC, neuron_idx, j) for j in range(n)]
    max_ = max(list_dist)
    result = [distance(PC, neuron_idx, i)/max_ for i in range(n)]
    return result

def plot_distance(PC, neuron_idx, rows, cols):

    my_list = list_distance(PC, neuron_idx)
    plot_distrib(my_list, "distances")
    show()

    plot(PC.x[neuron_idx] / meter, PC.y[neuron_idx] / meter, 'o',
                mfc='none', label=str(neuron_idx))
    color = 'b'

    for i in range(rows*cols):
        plot(PC.x[i] / meter,PC.y[i] / meter, color + '.', alpha= my_list[i])

    xlim(-10, rows)
    ylim(0, cols)
    xlabel('x')
    ylabel('y', rotation='horizontal')
    axis('equal')

    title("Distance plot")
    show()


def plot_spike_times(PC, my_indices, rows, cols, l):

    color = 'r'
    norm_l = normalize(l)

    plot_distrib(norm_l, "normalized spiking times")
    show()

    for i in range(rows*cols):
        if l[i]<inf:
            plot(PC.x[i] / meter, PC.y[i] / meter, color + '.', alpha=1-norm_l[i])

    for i in range(1,len(my_indices)):
        plot(PC.x[my_indices[i]] / meter, PC.y[my_indices[i]] / meter, 's', color='r',
             alpha=1- norm_l[my_indices[i]], label=str(my_indices[i]))

    neuron_idx = my_indices[0]
    plot(PC.x[neuron_idx] / meter, PC.y[neuron_idx] / meter, '*',
              label=str(neuron_idx))

    first_spiking = np.argmin(norm_l)
    plot(PC.x[first_spiking] / meter, PC.y[first_spiking] / meter, 'x', label="first")

    #color_gradient = plot(PC.x[new_indices] / meter, PC.y[new_indices] / meter, color + '.', alpha=float(alpha_indices)/m)

    xlim(-10, rows)
    ylim(0, cols)
    xlabel('x')
    ylabel('y', rotation='horizontal')
    axis('equal')
    title("Spiking time of cells")
    legend()
    show()

    return first_spiking

def normalize(l):
    norm_l = [i for i in l if i < inf]

    min_ = min(norm_l)
    length = max(norm_l) - min_
    result = []
    for i in l:
        if i < inf:
            result.append((i-min_)/length)
        else:
            result.append(i)
    return result

def sgmd(x):
    """Sigmoid (logistic) function."""

    return 1 / (1 + np.exp(-x))

def spiking_times_fun(MM, rows, cols, threshold):
    spiking_times = []

    print(len(MM.v))

    for element in range(rows * cols):
        if len(list(np.where(MM.v[element] >= threshold))[0]) > 0:
            spike_time = list(np.where(MM.v[element] >= threshold))[0].tolist()[0]
            spiking_times.append(spike_time)
        else:
            spiking_times.append(nan)


    return spiking_times
