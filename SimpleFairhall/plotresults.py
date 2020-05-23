from runresults import*

if False:
    plot_distance(PC,neuron_idx, rows, cols)


my_indicies = some_indices(neuron_idx)

first_spiking, last_spiking = plot_spike_times(PC, my_indicies, rows, cols, spiking_times)
show()

linePC0, =  plot(MPC.t / ms, MPC.v[0], label='PC' + str(neuron_idx))
linePC1, = plot(MPC.t / ms, MPC.v[1], label='PC' + str(my_indicies[1]))

linePC3, = plot(MPC.t / ms, MPC.v[3], label='PC' + str(my_indicies[3]))
linePC4, = plot(MPC.t / ms, MM.v[first_spiking], label='first')

linePC4, = plot(MPC.t / ms, MM.v[last_spiking], label='last')

savefig('Voltage of Pyramidal cells.png')
legend()
title("Voltage of Pyramidal cells")
show()

#plot_distance(PC, 0, rows, cols)

plot_connectivity(PC, SPC, rows, cols, "PC[" +str(50)+ "]" , 50)
show()
savefig('RecurrentconnectivityPC.png')

plot_connectivity(PC, SPCG, rows, cols, "G inputs")
show()
savefig('PC_Gconnectivity.png')

plot_connectivity(PC, SS, rows, cols, "S inputs")
show()
savefig('PC_Sconnectivity.png')

#Inhibitory
lineINH, = plot(MINH.t / ms, MINH.v[0], label='MINH')

lineG, = plot(MG.t / ms, MG.v[0], label='MG')
legend()
title("VoltageG_Inh_inputs")

show()

savefig('VoltageG_Inhcells.png')
