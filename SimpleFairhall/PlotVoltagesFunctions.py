from brian2 import*
import ConvertingFunctions

################### plot_votages_PC#####################

def plot_voltages_PC(modelClass, new_indices : list =[]):
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

    if len(new_indices) > 0:
        my_indices = new_indices
    else:
        my_indices = modelClass.my_indices

    for i in range(len(my_indices)):
        my_plot = plot(modelClass.MPC.t / ms, modelClass.MPC.v[i], label='PC' + str(my_indices[i]))
        index = my_indices[i]
        if index in list(modelClass.spikemon.i):
            plot(modelClass.PC_all_values['t'][index] / ms, modelClass.PC_all_values['v'][index], 'o', color=my_plot[0].get_color())

    if modelClass.plot_last_first:
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

def plot_voltages_mean_other_types(modelClass, type_list=['INH', 'threshold', 'weights', 'PC']):
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
            mean_threshold = np.mean([modelClass.Mthreshold[i].h for i in modelClass.Mthreshold.record], axis=0)
            plot(modelClass.Mthreshold.t / ms, mean_threshold, label='Threshold for mean PC cells.')

            # plot(modelClass.MthresholdG.t / ms, modelClass.MthresholdG.h[0], label='Threshold for G cell.')
        elif type_ == 'weights':
            if modelClass.model == "ThresholdModel":
                mean_weight_inside = np.mean([modelClass.weights[i].w for i in modelClass.weights.record], axis=0)
                plot(modelClass.weights[0].t/ ms, mean_weight_inside, label='Mean for weights(Poisson,PC)')

            if modelClass.model == "FairhallModel":

                mean_weight_inside = np.mean([modelClass.weights[i].w for i in modelClass.weights.record], axis=0)
                plot(modelClass.weights[0].t/ ms, mean_weight_inside, label='Mean for weights(Poisson,PC) IN trajectory')


                mean_weight2_inside = np.mean([modelClass.weights2[i].w for i in modelClass.weights2.record], axis=0)
                plot(modelClass.weights2[0].t/ ms, mean_weight2_inside, label='Mean for weights(Poisson,PC) OUT trajectory.')

        elif type_ == 'INH':

            mean_inh_threshold = np.mean([modelClass.MINH[i].v for i in modelClass.MINH.record], axis=0)
            plot(modelClass.MINH.t / ms, mean_inh_threshold, label='Inhibitory mean cells.')

        elif type_ == 'N':

            mean_noise = np.mean([modelClass.MN[i].v for i in modelClass.MINH.record], axis=0)
            plot(modelClass.MN.t / ms, mean_noise, label='Noise mean cells.')

        elif type_ == 'R':

            mean_random = np.mean([modelClass.MR[i].v for i in modelClass.MR.record], axis=0)
            plot(modelClass.MR.t / ms, mean_random, label='Random mean cells.')

        elif type_ == 'PC':
            if modelClass.plot_last_first:
                mean_PC = np.mean([modelClass.MM[i].v for i in modelClass.MM.record], axis=0)
                plot(modelClass.MM.t / ms, mean_PC, label='PC mean cells.')
            else:
                mean_PC = np.mean([modelClass.MPC[i].v for i in modelClass.MPC.record], axis=0)
                plot(modelClass.MPC.t / ms, mean_PC, label='PC mean cells.')

        else:
            raise ValueError("Type must be 'PC', 'R' , 'INH', 'threshold' or 'N' ")

    title(my_title)
    xlabel("Time in ms")
    ylabel("Voltage", rotation='vertical')
    legend()
    show()

def plot_voltages_mean_trajectory(modelClass):
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
    indices_trajectory = ConvertingFunctions.convert_list_to_threshold(modelClass.p['trajectory'][0, :], 1, 0)

    mean_PC_traj = np.mean([modelClass.MM[i].v for i in modelClass.MM.record if indices_trajectory[i] == 1], axis=0)
    plot(modelClass.MM.t / ms, mean_PC_traj, label='PC mean cells trajectory.')

    mean_PC_out_traj = np.mean([modelClass.MM[i].v for i in modelClass.MM.record if indices_trajectory[i] == 0], axis=0)
    plot(modelClass.MPC.t / ms, mean_PC_out_traj, label='PC mean cells outside trajectory.')

    title(my_title)
    xlabel("Time in ms")
    ylabel("Voltage", rotation='vertical')
    legend()
    show()

def verify_indices(modelClass):

    indices_trajectory = ConvertingFunctions.convert_list_to_threshold(modelClass.p['trajectory'][0, :], 1, 0)
    for i in range(len(indices_trajectory)):
        if indices_trajectory[i] == 0:
            plot(modelClass.PC.x[i] / meter, modelClass.PC.y[i]/meter, 'k' + '.')
    title("Outside trajectory")
    xlabel("meter")
    ylabel("meter", rotation='vertical')
    show()

    n = modelClass.p['rows'] * modelClass.p['cols'] - 1
    plot(modelClass.PC.x[n] / meter, modelClass.PC.y[n] / meter,'w', alpha=0.)
    plot(modelClass.PC.x[0] / meter, modelClass.PC.y[0] / meter, 'w', alpha=0.)

    for i in range(len(indices_trajectory)):
        if indices_trajectory[i] == 1:
            plot(modelClass.PC.x[i] / meter, modelClass.PC.y[i]/meter, 'k' + '.')
    title("Inside trajectory")
    xlabel("meter")
    ylabel("meter", rotation='vertical')
    show()
    return indices_trajectory

def plot_voltages_other_types(modelClass, type_list=['R','N', 'INH'], my_indices=[0]):
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
            plot_voltages_PC(modelClass,new_indices=my_indices)

        else:
            raise ValueError("Type must be 'PC', 'R' , 'INH', 'threshold' or 'N' ")

    title(my_title)
    xlabel("Time in ms")
    ylabel("Voltage", rotation='vertical')
    if len(my_indices) <= 5 :
        legend()
    show()
