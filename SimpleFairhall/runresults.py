from FirstModel import*
# To do: Find better way to record spike moments

neuron_idx = 50
my_indicies = some_indices(neuron_idx)
MPC = StateMonitor(PC, 'v', record=my_indicies)
MM = StateMonitor(PC, 'v', record=True)

spikemon = SpikeMonitor(PC, variables='v', record=True)


MINH = StateMonitor(INH, 'v', record=0)
spikemoninh = SpikeMonitor(INH, record=10)

run(100 * ms)


spiking_times = spiking_times_fun(MM, rows, cols, -36.1)
