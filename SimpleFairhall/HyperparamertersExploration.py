from brian2 import*
from MainFunctions import plot_run
########### Hyperparameters #################

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

