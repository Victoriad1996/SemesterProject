import matplotlib.pyplot as plt
from functions import plot_distrib, some_indices, spiking_times_fun
from brian2 import*



class FirstModel:
    def __init__(self, param):
        #start_scope()
        
        self.p = self._complete_parameters(param)
        self._param_sanity_checks(self.p)

        self._init_pyramidal_neurons()
        self._init_inhibitory_neurons()
        self._init_tonic_input_neurons()
        self._init_external_input_neurons()
        self._init_synapses()

    # updates parameters of model with p a dictionary of parameters
    def _complete_parameters(self, p):
        genericParam = {
            'rows' : 20,
            'cols' : 30,

            # Excitatory neurons
            "v_init_exc" : -60,         # Initial value of potential
            "v_leak_exc" : -68,         # Leak potential
            "v_reset_exc" : -60,        # Reset potential
            "v_thr_exc" : -36,          # Spiking threshold
            'tau_dyn_exc' : 50 * ms,    # Leak timescale
            "tau_refr_exc" : 8 * ms,    # Refractory period
            "r_max_exc" : 20 * Hz,      # Maximum rate
            "lambda_PL_exc" : 0.15 * metre,    # ???

            # Inhibitory neurons
            "v_leak_inh" : -80,         # Leak potential
            "v_reset_inh": -80,         # Reset potential
            "v_thr_inh" : -50,          # Spiking threshold
            'tau_dyn_inh': 5 * ms,      # Leak timescale
            "tau_refr_inh" : 2 * ms,    # Refractory period
            "gi_inh" : 1,               # ???

            # Tonic Neurons
            "v_reset_tonic" : -80,        # Leak potential
            "v_leak_tonic" : -80,         # Reset potential
            "v_thr_tonic" : -50,          # Spiking threshold
            'tau_dyn_tonic': 5 * ms,      # Leak timescale
            "tau_refr_tonic" : 2 * ms,    # Refractory period
            "gi_tonic" : 1,               # ???

            # External Input Neurons
            "v_reset_ext": -80,  # Leak potential
            "v_leak_ext": -80,  # Reset potential
            "v_thr_ext": -50,  # Spiking threshold
            'tau_dyn_ext': 5 * ms,  # Leak timescale
            "tau_refr_ext": 2 * ms,  # Refractory period
            "gi_ext": 1,  # ???

            # Synapses
            "w_PC" : 0.5,           # Recurrent weight from PC to PC
            "w_PCINH" : 0.7,        # Weight PC to INH
            "w_INHPC" : 0.5,        # Weight INH to PC
            "w_GPC" : 0.9,          # Weight G to PC
            "w_SPC" : 0.8           # Weigth S to PC
        }

        pnew = {}
        for k, v in genericParam.items():
            if k not in p.keys():
                pnew[k] = v
            else:
                pnew[k] = p[k]

        return pnew


    def _param_sanity_checks(self, p):
        pass


    # Initialises the pyramidal neurons as a Brian2 NeuronGroup.
    def _init_pyramidal_neurons(self):
        eqs_exc = '''
            dv/dt = (I - (v - v_leak_exc)) / tau : 1
            x : metre
            y : metre
            tau : second
            I : 1
            '''

        self.PC = NeuronGroup(self.p['rows'] * self.p['cols'], eqs_exc, threshold='v>v_thr_exc', reset='v = v_reset_exc', refractory=self.p["tau_refr_exc"], method='euler')

        # initialize the grid positions
        rows = self.p['rows']
        grid_dist = 4 * meter
        self.PC.tau = self.p['tau_dyn_exc']
        # x and y position on the grid of the cells.
        self.PC.x = '(i // rows) * grid_dist'
        self.PC.y = '(i % rows) * grid_dist'
        #Initialises voltage
        self.PC.v = self.p['v_init_exc']
        self.PC.I = '0'


    # Initialises the inhibitory neurons as a Brian2 NeuronGroup.
    def _init_inhibitory_neurons(self):
        eqs_inh = '''
        dv/dt = (I - (v - v_leak_inh)) / tau : 1
        tau : second
        I : 1
        '''

        self.INH = NeuronGroup(int(self.p['rows'] * self.p['cols'] / 10), eqs_inh, threshold='v>v_thr_inh',
                               reset='v = v_reset_inh', refractory=self.p['tau_refr_inh'], method='euler')
        self.INH.I = 0
        self.INH.tau = self.p['tau_dyn_inh']


    # Initialises tonic input that will "draw" the trajectory.
    def _init_tonic_input_neurons(self):
        eqs_tonic = '''
        dv/dt = (I - (v - v_leak_tonic)) / tau : 1
        tau : second
        I : 1
        '''

        self.G = NeuronGroup(self.p['rows'], eqs_tonic, threshold='v > v_thr_tonic', reset='v = v_reset_tonic',
                             refractory=self.p['tau_refr_tonic'], method='euler')
        self.G.I = -10
        self.G.tau = self.p['tau_dyn_tonic']


    # Initialises external input. It should spread the spikes along the trajectory.
    def _init_external_input_neurons(self):
        eqs_ext_inp = '''
        dv/dt = (I - (v - v_leak_ext)) / tau : 1
        tau : second
        I : 1
        '''

        self.S = NeuronGroup(1, eqs_ext_inp, threshold='v > v_thr_ext', reset='v = v_reset_ext',
                             refractory=self.p['tau_refr_ext'], method='euler')
        self.S.I = -10
        self.S.tau = self.p['tau_dyn_ext']


    # Initialises all the synapses
    def _init_synapses(self):
        # Needed for brian2 equations
        rows = self.p['rows']
        cols = self.p['cols']

        ###########################
        # EXC-EXC synapses
        ###########################
        self.SPC = Synapses(self.PC, self.PC, 'w:1', on_pre= 'v_post += w')
        #self.SPC.connect('i // rows - j // rows == 1 or i // rows - j // rows == - 1 ')

        self.SPC.connect('(( i // rows - j // rows)**2 + ( i % cols -  j % cols)**2 )< 4')
        self.SPC.w = '2*exp(-((x_pre-x_post)**2+(y_pre - y_post)**2)/(2*15*metre)**2)'


        ###########################
        # EXC-INH and INH-EXC synapses
        ###########################
        self.SPCINH = Synapses(self.INH, self.PC, on_pre='v_post-=0.01')
        self.SPCINH.connect(p=0.)

        self.SINHPC = Synapses(self.PC, self.INH, on_pre='v_post+=0.01')
        self.SINHPC.connect(p=0.)

        ###########################
        # TONIC-EXC synapses
        ###########################
        #Draws the trajectory
        eqs_spcg = '''
        ((j % rows <= 17 and j%rows >= 15) and (j // cols <= 10 )) or
        ((j // cols >= 8 and j // cols <= 10)
            and (j % rows <= 15 and j%rows >= 4)
            or ((j//cols>= 10) and (j%rows >=4 and j%rows<=6))
        )
        '''

        self.SPCG = Synapses(self.G, self.PC, on_pre='v_post+=0.1')
        self.SPCG.connect(eqs_spcg)

        ###########################
        # EXTERNAL INP-EXC synapses
        ###########################
        self.SS = Synapses(self.S, self.PC, on_pre='v_post+=0.5')
        # Triggers a few neurons in the trajectory.
        self.SS.connect('((j % rows <= 17 and j%rows >= 15) and (j // cols == 0 ))')


    # Plots the distribution of the distances of pyramidal cells to the neuron[index].
    def plot_neuron_distances(self, index):
        distances_list = []
        for i in range(1, self.p['rows']):
            distances_list.append(0.5*exp(-((self.PC.x[index]-self.PC.x[i])**2+(self.PC.y[index] - self.PC.y[i])**2)/(2*15*metre)**2))
        plot_distrib(distances_list, "Distances to " + str(index) + "th neuron")


    # TODO: Find better way to record spike moments
    def run(self):

        # Gets some indices. Needed for having points references when plotting spiking times,
        # and when plotting the voltages of some neurons.
        self.neuron_idx = 50
        self.my_indicies = some_indices(self.neuron_idx)

        # StateMonitor will record the voltages values of indices given to record.

        # Records only for a few indices.
        self.MPC = StateMonitor(self.PC, 'v', record=self.my_indicies)
        #Records for all indices, used for recording spiking times.
        # TODO: Find better way to record spike moments
        self.MM = StateMonitor(self.PC, 'v', record=True)
        # Records the G inputs of one cell.
        self.MG = StateMonitor(self.G, 'v', record=0)
        # Records the inhibitory of one cell.
        self.MINH = StateMonitor(self.INH, 'v', record=0)


        # Records the spikes of Pyramidal cells.
        self.spikemon = SpikeMonitor(self.PC, variables='v', record=True)
        # Records the spikes of one inhibitory cell.
        self.spikemoninh = SpikeMonitor(self.INH, record=10)

        netObjs = {k: v for k, v in vars(self).items() if isinstance(v, BrianObject)}
        net = core.network.Network(netObjs)
        net.run(1 * second, namespace=self.p)

        # # For debugging reasons.
        # print("get objects in namespace")
        # print(core.magic.get_objects_in_namespace(level=0))
        #
        # #Useful to check what "neurongroup" they have been registered as, and if it matches with the one that will be used for run()
        # print("self PC ", self.PC)
        # print("self INH ", self.INH)
        # print("self G ", self.G)
        # print("self S ", self.S)
        #
        # print("self MPC ", self.MPC)
        # print("self MM ", self.MM)
        # print("self MPC ", self.MPC)
        # print("self MG ", self.MG)
        # print("self MINH ", self.MINH)
        #
        #
        # # Prints variables that will be used for the run()
        # print("collect ")
        # collect()
        # print(collect())
        #
        # # Run, report serves
        # run(100 * ms, report='stdout', report_period=10*ms)
        #

        # Plots the voltages of recorded pyramidal cell.
        plot(self.MPC.t / ms, self.MPC.v[0], label="0")
        title("MPC")
        show()

        # Plots the voltages of recorded pyramidal cell.
        plot(self.MPC.t / ms, self.MM.v[0], label="MM 0")
        title("MM 0")
        show()

        # Following line does not run because, the recorded voltages are empty.
        # self.spiking_times = spiking_times_fun(self.MM, self.p['rows'], self.p['cols'], -36.1)
        # print("self.spikingtime ", self.spiking_times[0])


