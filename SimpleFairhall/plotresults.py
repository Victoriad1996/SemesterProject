from runresults import*

if False:
    plot_distance(PC,neuron_idx, rows, cols)


my_indicies = some_indices(neuron_idx)

first_spiking = plot_spike_times(PC, my_indicies, rows, cols, spiking_times)


linePC0, =  plot(MPC.t / ms, MPC.v[0], label='PC' + str(neuron_idx))
linePC1, = plot(MPC.t / ms, MPC.v[1], label='PC' + str(my_indicies[1]))

linePC3, = plot(MPC.t / ms, MPC.v[3], label='PC' + str(my_indicies[3]))
linePC4, = plot(MPC.t / ms, MM.v[first_spiking], label='first')
#lineINH, = plot(MINH.t / ms, MINH.v[0], label='MINH')

legend()
title("Voltage of Pyramidal cells")
show()

#Inhibitory
if False :
    lineINH, = plot(MINH.t / ms, MINH.v[0], label='MINH')
    legend(handles=[lineINH])
    title("Voltage of Inhibitory cells")

    show()
