#%%

from FirstModelOOP import FirstModel
from brian2 import*
import functions



#%%

params = {
    "rows" : 20,
    "cols" : 25
}

fm1 = FirstModel(params)

#%%

fm1.plot_neuron_distances()

#%%

functions.visualise_connectivity(fm1.SPC, "Recurrent connectivity")
functions.visualise_connectivity(fm1.SPCINH, "Connectivity PC and Inhibitory")
functions.visualise_connectivity(fm1.SPCG, "Recurrent connectivity PC and G")


#%%

ie = 60
ei = -80
vth_i = -50
tr_i = 2 * ms
gi = 1

eqs_ext_inp = '''
dv/dt = (I-(v-ei) + ie)/tau : 1
tau : second
I : 1
'''

N = NeuronGroup(1, eqs_ext_inp, threshold='v>vth_i', reset='v = ei', refractory=tr_i, method='euler')
N.I = - 10
N.tau = 5 * ms

MN = StateMonitor(N, 'v', record = True)

#%%

start_scope()
fm1.run()

#%%

run(100*ms)
plot(MN.t /ms, MN.v[0])
title('Test recording')

#%%

if False:
    plot_distance(PC,neuron_idx, rows, cols)

first_spiking, last_spiking = functions.plot_spike_times(fm1)

#%%

linePC0, =  plot(fm1.MPC.t / ms, fm1.MPC.v[0], label='PC' + str(fm1.neuron_idx))
linePC1, = plot(MPC.t / ms, MPC.v[1], label='PC' + str(my_indicies[1]))

linePC3, = plot(MPC.t / ms, MPC.v[3], label='PC' + str(my_indicies[3]))
linePC4, = plot(MPC.t / ms, MM.v[first_spiking], label='first')

linePC4, = plot(MPC.t / ms, MM.v[last_spiking], label='last')

savefig('Voltage of Pyramidal cells.png')
legend()
title("Voltage of Pyramidal cells")
show()

#%%

#plot_distance(PC, 0, rows, cols)

plot_connectivity(fm1.PC, fm1.SPC, fm1.p["rows"], fm1.p["cols"], "PC[" +str(50)+ "]" , 50)
show()

#%%

plot_connectivity(fm1.PC, fm1.SPCG, fm1.p["rows"], fm1.p["cols"], "G inputs")
savefig('PC_Gconnectivity.png')
show()

#%%

plot_connectivity(fm1.PC, fm1.SS, fm1.p["rows"], fm1.p["cols"], "S inputs")
savefig('PC_Sconnectivity.png')
show()

#%%

#Inhibitory
lineINH, = plot(fm1.MINH.t / ms, fm1.MINH.v[0], label='MINH')

lineG, = plot(fm1.MG.t / ms, fm1.MG.v[0], label='MG')
legend()
title("VoltageG_Inh_inputs")

savefig('VoltageG_Inhcells.png')
show()

#%%

x = 1

if x != 0:
    raise ValueError("Expected zero number of links, got", x)
