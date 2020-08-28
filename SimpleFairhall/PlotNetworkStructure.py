from brian2 import*
import BasicFunctions


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

    BasicFunctions.plot_distrib(list1, "2 - 2 * exp(-||n1 - n2||_2)")
    BasicFunctions.plot_distrib(list2, "||n1 - n2||_2")


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
    my_list = BasicFunctions.list_distance(modelClass.PC, modelClass.neuron_idx)
    # Plots the distribution of this list of distances.
    BasicFunctions.plot_distrib(my_list, "distances")

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
