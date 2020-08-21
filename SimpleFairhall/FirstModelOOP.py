from brian2 import*
import functions

class FirstModel:
    def __init__(self, param, model="Fairhall"):
        self.model = model
        self.reshaped_spikemon = None
        self.variance_spike_times = None
        self.mean_spike_times = None
        self.number_spiking_cells = None
        self.number_spiking_outside_trajectory = None
        self.variance_spikes_trajectory = None
        self.mean_spikes_trajectory = None
        self.p = self._complete_parameters(param)
        self.has_run = False
        self.delay = None
        self.PC_spiking_times = None
        self.INH_spiking_times = None

        self._param_sanity_checks(self.p)
        self._init_pyramidal_neurons()
        self._init_inhibitory_neurons()
        self._init_standard_synapses()
        if self.model=="Fairhall":
            self._init_external_input_neurons()
            self._init_external_input_synapses()
            if 'result_G' in self.p.keys():
                self.G_spiking_times = None
                self._init_tonic_input_neurons()
                self._init_tonic_synapses()
        else:
            self._init_random_input()
            self._init_random_synapses()

            self.G_spiking_times = None
            self._init_tonic_input_neurons()
            self._init_tonic_synapses()

        self.neuron_idx = 50
        self.my_indices = functions.some_indices(self.neuron_idx)

        # For plotting purposes
        self.excitatory_matrix = np.zeros([self.p['rows'], self.p['cols']])

    # Print the parameters. Helps for debugging
    def print_param(self, type_list=['INH']):
        for type_ in type_list:
            if type_ == 'INH':
                print('INH : ')
                print("v_leak_inh ", self.p["v_leak_inh"])
                print("v_reset_inh ", self.p["v_reset_inh"])
                print("v_thr_inh ", self.p["v_thr_inh"])
                print("tau_dyn_inh ", self.p["tau_dyn_inh"])
                print("tau_refr_inh ", self.p["tau_refr_inh"])
            elif type_ == 'S':
                print('S : ')
                print("v_reset_ext ", self.p["v_reset_ext"])
                print("v_leak_ext ", self.p["v_leak_ext"])
                print("v_thr_ext ", self.p["v_thr_ext"])
                print("tau_dyn_ext ", self.p["tau_dyn_ext"])
                print("tau_refr_ext ", self.p["tau_refr_ext"])

    def set_delay(self, new_delay):
        self.delay = new_delay

    def reset_neuron_idx(self, neuron_idx):
        self.neuron_idx = neuron_idx

    def reset_neuron_idx(self, list_neurons):
        self.my_indices = list_neurons

    # updates parameters of model with p a dictionary of parameters
    def _complete_parameters(self, p):
        genericParam = {
            'rows' : 20,
            'cols' : 30,
            "thresh_step": 0.2,

            # Excitatory neurons
            "v_init_exc" : -60,         # Initial value of potential
            "v_leak_exc" : -68,         # Leak potential
            "v_reset_exc" : -60,        # Reset potential
            "v_thr_exc" : -36,          # Spiking threshold
            'tau_dyn_exc' : 50 * ms,    # Leak timescale
            "tau_refr_exc" : 8 * ms,    # Refractory period
            "r_max_exc" : 20 * Hz,      # Maximum rate
            'rec_weight' : 7,         # Recurrent weight
            "lambda_PL_exc" : 0.15 * metre,    # ???

            # Inhibitory neurons
            "num_inhib_neurons" : 10,
            "v_leak_inh" : -10,         # Leak potential
            "v_reset_inh": -80,         # Reset potential
            "v_thr_inh" : -50,          # Spiking threshold
            'tau_dyn_inh': 50 * ms,      # Leak timescale
            "tau_refr_inh" : 2 * ms,    # Refractory period
            "gi_inh" : 1,               # ???
            "inh_weight_pi" : 0.02,
            "inh_weight_ip" : 0.01,
            #
            # # Tonic Neurons
            "num_tonic_neurons" : 20, # Number of tonic neurons
            "v_leak_tonic": -30,          # Leak potential
            "v_reset_tonic" : -80,        # Reset potential
            "v_thr_tonic" : -50,          # Spiking threshold
            'tau_dyn_tonic': 5 * ms,      # Leak timescale
            "tau_refr_tonic" : 2 * ms,    # Refractory period
            "gi_tonic" : 1,               # ???
            "tonic_weight" : 0.03,         # Tonic weight

            # External Input Neurons
            "num_ext_neurons" : 10, # Number of external input neurons
            "v_leak_ext": -40,           # Leak potential
            "v_reset_ext": -80,          # Reset potential
            "v_thr_ext": -50,            # Spiking threshold
            'tau_dyn_ext': 5 * ms,       # Leak timescale
            "tau_refr_ext": 2 * ms,      # Refractory period
            "gi_ext": 1,  # ???
            "ext_weight" : 0.5,

            # Random inputs
            "num_random_neurons" : 20,
            "v_leak_random" : -30, # Leak potential
            "v_reset_random" : -80,        # Reset potential
            "v_thr_random": -50,  # Spiking threshold
            "input_rate" : 50*Hz, # Rate of spikes for the
            'R_weight' : 15,
            "tau_dyn_random" : 5 * ms,      # Leak timescale

            # Synapses
            "w_PC" : 0.5,           # Recurrent weight from PC to PC
            "w_PCINH" : 0.7,        # Weight PC to INH
            "w_INHPC" : 0.5,        # Weight INH to PC
            "w_GPC" : 0.9,          # Weight G to PC
            "w_SPC" : 0.8           # Weigth S to PC
        }

        pnew = p
        for k, v in genericParam.items():
            if k not in p.keys():
                pnew[k] = v
            else:
                pnew[k] = p[k]

        return pnew

    def _param_sanity_checks(self, p):
        pass

    # If want to modify parameters after creating the class, before running
    def modify_param(self, p):
        if self.has_run:
            raise Warning("The network already ran thus the change of parameters should have no impact")

        for element in p.keys():
            self.p[element] = p[element]

    # Initialises the pyramidal neurons as a Brian2 NeuronGroup.
    def _init_pyramidal_neurons(self):

        #v_leak_exc
        eqs_exc = '''
            dv/dt = (- (v - I)) / tau : 1
            h : 1
            I : 1
            thresh_step : 1
            x : metre
            y : metre
            tau : second
            '''

        if self.model == 'Fairhall':
            #If in case of Fairhall model, the threshold does not adapt.
            #v_reset_exc
            self.PC = NeuronGroup(self.p['rows'] * self.p['cols'], eqs_exc, threshold='v>h', reset='v = I',
                                  refractory=self.p["tau_refr_exc"], method='euler')
        else:
        # Here the threshold lowers if the neuron spikes
            self.PC = NeuronGroup(self.p['rows'] * self.p['cols'], eqs_exc, threshold='v>h',
                                  reset='v = v_reset_exc; h = h - 10', refractory=self.p["tau_refr_exc"],
                                  method='euler')
        #thresh_step
            self.PC.thresh_step = self.p['thresh_step']

        # initialize the grid positions

        # Variables important for Brian purposes (the equations are written with strings).
        rows = self.p['rows']
        grid_dist = 4 * meter
        self.PC.tau = self.p['tau_dyn_exc']
        # x and y position on the grid of the cells.
        self.PC.x = '(i // rows) * grid_dist'
        self.PC.y = '(i % rows) * grid_dist'
        #Initialises voltage
        self.PC.v = self.p['v_init_exc']
        #self.PC.h = functions.convert_matrix_to_threshold(self.p['connection_matrix_S_fairhall'][0,:],self.p['rows'], self.p['cols'],self.p['v_thr_exc'] - 20, self.p['v_thr_exc'])
        self.PC.h = functions.convert_list_to_threshold(self.p['connection_matrix_S'][0,:], self.p['v_thr_exc'] -20 , self.p['v_thr_exc'])
        self.PC.I = functions.convert_list_to_threshold(self.p['connection_matrix_S'][0, :], -45,-68)

    # Initialises the inhibitory neurons as a Brian2 NeuronGroup.
    def _init_inhibitory_neurons(self):
        eqs_inh = '''
        dv/dt = ( - (v - v_leak_inh)) / tau : 1
        tau : second
        '''
        #int(self.p['rows'] * self.p['cols'] / 10)
        self.INH = NeuronGroup(int(self.p['rows'] * self.p['cols'] / 10), eqs_inh, threshold='v>v_thr_inh',
                               reset='v = v_reset_inh', refractory=self.p['tau_refr_inh'], method='euler')
        self.INH.tau = self.p['tau_dyn_inh']
        self.INH.v = self.p['v_reset_inh']


    # Initialises tonic input that will "draw" the trajectory.
    def _init_tonic_input_neurons(self):

        eqs_tonic = '''
        dv/dt = ( - h * (v - v_leak_tonic)) / tau : 1
        dh/dt = (- 0.5 * h) / tau : 1
        
        tau : second
        '''

        self.G = NeuronGroup(self.p['num_tonic_neurons'], eqs_tonic, threshold='v>v_thr_tonic',
                             reset='v = v_reset_tonic',
                             refractory=self.p['tau_refr_tonic'], method='euler')
        self.G.h = 10
        self.G.tau = self.p['tau_dyn_tonic']
        self.G.v = -80

    def _init_random_input(self):
        #self.R = PoissonGroup(self.p['num_tonic_neurons'],rates=self.p['input_rate'])

        eqs_tonic = '''
        dv/dt = ( - (v - v_leak_random)) / tau : 1
        tau : second
        '''

        self.R = NeuronGroup(self.p['num_random_neurons'], eqs_tonic, threshold='v>v_thr_random', reset='v = v_reset_tonic',
                             refractory=50*ms, method='euler')
        self.R.tau = self.p['tau_dyn_random']
        self.R.v = -80


    # Initialises external input. It should spread the spikes along the trajectory.
    def _init_external_input_neurons(self):
        eqs_ext_inp = '''
        dv/dt = ( - h * (v - v_leak_ext)) / tau : 1
        
        dh/dt = (- h) / tau : 1
        tau : second
        '''

        #
        self.S = NeuronGroup(self.p['num_ext_neurons'], eqs_ext_inp, threshold='v>v_thr_ext', reset='v = v_reset_ext',
                             refractory=self.p['tau_refr_ext'], method='euler')
        self.S.tau = self.p['tau_dyn_ext']
        self.S.h = 10
        self.S.v = -80

    # Initialises all the synapses
    def _init_standard_synapses(self):
        # Needed for brian2 equations
        rows = self.p['rows']
        cols = self.p['cols']
        weight = self.p['rec_weight']
        ###########################
        # EXC-EXC synapses
        ###########################
        self.SPC = Synapses(self.PC, self.PC, 'w:1', on_pre= 'v_post += w')
        self.SPC.connect('(( i // rows - j // rows)**2 + ( i % rows -  j % rows)**2 )< 4')
        #self.SPC.connect()
        self.SPC.w = 'weight*exp(-((x_pre-x_post)**2+(y_pre - y_post)**2)/(30*metre)**2)'

        # If already specified the delay to put, don't modify it.
        if self.delay!= None:
            self.SPC.delay = self.delay
        else:
            #self.SPC.delay = '(2 - 2*exp(-(abs(x_pre-x_post)+abs(y_pre - y_post))/(30*metre))) * second'
            self.SPC.delay = '(((x_pre-x_post)**2+(y_pre - y_post)**2)/(50*metre)**2) * second'


        ###########################
        # EXC-INH and INH-EXC synapses
        ###########################
        self.SPCINH = Synapses(self.INH, self.PC, 'w:1', on_pre='v_post-=w')
        self.SPCINH.connect(p=0.5)
        self.SPCINH.w = self.p["inh_weight_pi"]

        self.SINHPC = Synapses(self.PC, self.INH, 'w:1', on_pre='v_post+=w')
        self.SINHPC.connect(p=0.5)

        self.SINHPC.w = self.p["inh_weight_ip"]

    def _init_tonic_synapses(self):

        """

                self.SPCG = Synapses(self.G, self.PC, 'w:1',on_pre='v_post+=w')
                if 'connection_matrix_G' in self.p.keys():
                    sources, targets = functions.convert_matrix_to_source_target(self.p['connection_matrix_G'])
                    self.SPCG.connect(i=sources, j=targets, p=0.)
                else:
                    eqs_spcg = '''
                            ((j % rows <= 17 and j%rows >= 15) and (j // rows <= 10 ))
                            or ((j // rows >= 8 and j // rows <= 10) and (j % rows <= 15 and j % rows >= 4))
                            or ((j//rows>= 10) and (j % rows >=4 and j % rows<=6))
                            '''
                    self.SPCG.connect(eqs_spcg)

                self.SPCG.w = self.p['tonic_weight']
        """
        ###########################
        # TONIC-EXC synapses
        ###########################
        #Draws the trajectory

        self.SPCG = Synapses(self.G, self.PC, 'w:1',on_pre='v_post+=w')
        self.SPCG.connect(p=0.5)
        self.SPCG.w = self.p['tonic_weight']


    def _init_external_input_synapses(self):

        ###########################
        # EXTERNAL INP-EXC synapses
        ###########################
        self.SS = Synapses(self.S, self.PC, 'w:1', on_pre='v_post+=w')
        # Triggers a few neurons in the trajectory.
        #self.SS.connect('((j % rows <= 17 and j%rows >= 15) and (j // rows == 0 ))')
        if 'connection_matrix_S' in self.p.keys():
            sources, targets = functions.convert_matrix_to_source_target(self.p['connection_matrix_S'])
            # synapses = Synapses(G, G, on_pre='v_post += 0.2')
            #self.SS.connect(i=sources, j=targets, p=0.)
            self.SS.connect(p=0.)
        else:
            eqs_ss = '((j % rows >=4 and j % rows<=6) and (j // rows == 10 ))'
            self.SS.connect(eqs_ss)
        self.SS.w = self.p['ext_weight']
        #self.SS.connect('j % rows == 5 and j // rows == 10')

    def _init_random_synapses(self):
        ###########################
        # RANDOM - EXC synapses
        ###########################
        self.SRPC = Synapses(self.R, self.PC, 'w:1', on_pre='v_post+=w')
        self.SRPC.connect(p=0.01)
        self.SRPC.w = 15
        #self.SRPC.w = self.p['R_weight']

    # TODO: Find better way to record spike moments
    def run(self, duration=50*ms, show_PC=False, show_other=False):
        # If show_ = True, then plots voltages

        # StateMonitor will record the voltages values of indices given to record.

        # Records only for a few indices.
        self.MPC = StateMonitor(self.PC, 'v', record=self.my_indices)
        #Records for all indices, used for recording spiking times.
        # TODO: Find better way to record spike moments
        self.MM = StateMonitor(self.PC, 'v', record=True)

        if self.model == 'Fairhall':
            if 'result_G' in self.p.keys():
                # Records the G inputs of one cell.
                self.MG = StateMonitor(self.G, 'v', record=0)
                self.MthresholdG = StateMonitor(self.G, 'h', record=True)
                self.spikemong = SpikeMonitor(self.G, variables='v', record=True)

                self.G_all_values = self.spikemong.all_values()
        # Records the G inputs of one cell.
            self.MS = StateMonitor(self.S, 'v', record=0)
            self.spikemons = SpikeMonitor(self.S, variables='v', record=True)

            self.S_all_values = self.spikemons.all_values()
        else:

            self.spikemon_random = SpikeMonitor(self.R)
        # Records the inhibitory of one cell.
        self.MINH = StateMonitor(self.INH, 'v', record=0)

        #self.v_thres = StateMonitor(self.PC, 'v_leak_exc', record=0)
        self.Mthreshold = StateMonitor(self.PC, 'h', record=True)
        # Records the spikes of Pyramidal cells.
        self.spikemon = SpikeMonitor(self.PC, variables='v', record=True)
        # Records the spikes of one inhibitory cell.
        self.spikemoninh = SpikeMonitor(self.INH,variables='v', record=True)


        # TODO : Find a way to be able to run several times
        if not self.has_run:
            netObjs = {k: v for k, v in vars(self).items() if isinstance(v, BrianObject)}
            self.net = core.network.Network(netObjs)
            self.net.run(duration, namespace=self.p)
        else:
            self.net.run(duration, namespace=self.p)

        # For checking if the network ran
        self.has_run = True

        # Records spiking information about different
        self.PC_all_values = self.spikemon.all_values()
        self.INH_all_values = self.spikemoninh.all_values()


        # Following line does not run because, the recorded voltages are empty.
        #functions.spiking_times_fun(self, -36.1)

        functions.spiking_times_fun(self)

        if show_PC:
            functions.plot_voltages_PC(self)
            #my_plot = plot(self.MPC.t / ms, self.MPC.v, label='PC v_leak_exc' )
            #show()

        if show_other:
            functions.plot_voltages_other_types(self)

