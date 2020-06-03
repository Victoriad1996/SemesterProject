from functions import some_indices, spiking_times_fun, plot_voltages_PC
from brian2 import*
import functions



class FirstModel:
    def __init__(self, param):
        self.has_run = False
        self.delay = None
        self.PC_spiking_times = None
        self.G_spiking_times = None
        self.INH_spiking_times = None

        self.p = self._complete_parameters(param)
        self._param_sanity_checks(self.p)
        self._init_pyramidal_neurons()
        self._init_inhibitory_neurons()
        self._init_tonic_input_neurons()
        self._init_external_input_neurons()
        self._init_synapses()
        self.neuron_idx = 50
        self.my_indices = some_indices(self.neuron_idx)

    def print_param(self, type_list=['G', 'INH']):
        for type_ in type_list:
            if type_ == 'INH':
                print('INH : ')
                print("v_leak_inh ", self.p["v_leak_inh"])
                print("v_reset_inh ", self.p["v_reset_inh"])
                print("v_thr_inh ", self.p["v_thr_inh"])
                print("tau_dyn_inh ", self.p["tau_dyn_inh"])
                print("tau_refr_inh ", self.p["tau_refr_inh"])
            elif type_ == 'G':
                print('G : ')

                print("v_leak_tonic ", self.p["v_leak_tonic"])
                print("v_reset_tonic ", self.p["v_reset_tonic"])
                print("v_thr_tonic ", self.p["v_thr_tonic"])
                print("tau_dyn_tonic ", self.p["tau_dyn_tonic"])
                print("tau_refr_tonic ", self.p["tau_refr_tonic"])
            elif type_ == 'S':
                print('S : ')
                print("v_reset_ext ", self.p["v_reset_ext"])
                print("v_leak_ext ", self.p["v_leak_ext"])
                print("v_thr_ext ", self.p["v_thr_ext"])
                print("tau_dyn_ext ", self.p["tau_dyn_ext"])
                print("tau_refr_ext ", self.p["tau_refr_ext"])

    def print_delay(self):
        print(self.delay)

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
            "v_leak_inh" : -20,         # Leak potential
            "v_reset_inh": -80,         # Reset potential
            "v_thr_inh" : -50,          # Spiking threshold
            'tau_dyn_inh': 5 * ms,      # Leak timescale
            "tau_refr_inh" : 2 * ms,    # Refractory period
            "gi_inh" : 1,               # ???

            # Tonic Neurons
            "v_leak_tonic": -30,          # Leak potential
            "v_reset_tonic" : -80,        # Reset potential
            "v_thr_tonic" : -50,          # Spiking threshold
            'tau_dyn_tonic': 5 * ms,      # Leak timescale
            "tau_refr_tonic" : 2 * ms,    # Refractory period
            "gi_tonic" : 1,               # ???

            # External Input Neurons
            "v_leak_ext": -40,           # Leak potential
            "v_reset_ext": -80,          # Reset potential
            "v_thr_ext": -50,            # Spiking threshold
            'tau_dyn_ext': 5 * ms,       # Leak timescale
            "tau_refr_ext": 2 * ms,      # Refractory period
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

    def modify_param(self, p):
        for element in p.keys():
            self.p[element] = p[element]

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
        dv/dt = ( - (v - v_leak_inh)) / tau : 1
        tau : second
        '''
        #int(self.p['rows'] * self.p['cols'] / 10)
        self.INH = NeuronGroup(int(self.p['rows'] * self.p['cols'] / 10), eqs_inh, threshold='v>v_thr_inh',
                               reset='v = v_reset_inh', refractory=self.p['tau_refr_inh'], method='euler')
        self.INH.tau = self.p['tau_dyn_inh']


    # Initialises tonic input that will "draw" the trajectory.
    def _init_tonic_input_neurons(self):
        eqs_tonic = '''
        dv/dt = ( - (v - v_leak_tonic)) / tau : 1
        tau : second
        '''

        self.G = NeuronGroup(self.p['rows'], eqs_tonic, threshold='v>v_thr_tonic', reset='v = v_reset_tonic',
                             refractory=self.p['tau_refr_tonic'], method='euler')


        self.G.tau = self.p['tau_dyn_tonic']


    # Initialises external input. It should spread the spikes along the trajectory.
    def _init_external_input_neurons(self):
        eqs_ext_inp = '''
        dv/dt = ( - (v - v_leak_ext)) / tau : 1
        tau : second
        '''

        self.S = NeuronGroup(1, eqs_ext_inp, threshold='v>v_thr_ext', reset='v = v_reset_ext',
                             refractory=self.p['tau_refr_ext'], method='euler')
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
        #self.SPC.connect()
        self.SPC.connect('(( i // rows - j // rows)**2 + ( i % rows -  j % rows)**2 )< 40')
        #self.SPC.w = 1.
        #self.SPC.connect()
        self.SPC.w = '2*exp(-((x_pre-x_post)**2+(y_pre - y_post)**2)/(30*metre)**2)'


        if self.delay!= None:
            self.SPC.delay = self.delay
        else:
            #self.SPC.delay = '(2 - 2*exp(-(abs(x_pre-x_post)+abs(y_pre - y_post))/(30*metre))) * second'
            self.SPC.delay = '(((x_pre-x_post)**2+(y_pre - y_post)**2)/(50*metre)**2) * second'


        ###########################
        # EXC-INH and INH-EXC synapses
        ###########################
        self.SPCINH = Synapses(self.INH, self.PC, on_pre='v_post-=0.02')
        self.SPCINH.connect(p=0.)

        self.SINHPC = Synapses(self.PC, self.INH, on_pre='v_post+=0.01')
        self.SINHPC.connect(p=0.)

        ###########################
        # TONIC-EXC synapses
        ###########################
        #Draws the trajectory
        eqs_spcg = '''
        ((j % rows <= 17 and j%rows >= 15) and (j // rows <= 10 ))
        or ((j // rows >= 8 and j // rows <= 10) and (j % rows <= 15 and j % rows >= 4))
        or ((j//rows>= 10) and (j % rows >=4 and j % rows<=6))
        '''

        self.SPCG = Synapses(self.G, self.PC, on_pre='v_post+=0.17')
        #self.SPCG.connect(eqs_spcg)
        self.SPCG.connect(eqs_spcg)

        ###########################
        # EXTERNAL INP-EXC synapses
        ###########################
        self.SS = Synapses(self.S, self.PC, on_pre='v_post+=0.5')
        # Triggers a few neurons in the trajectory.
        #self.SS.connect('((j % rows <= 17 and j%rows >= 15) and (j // rows == 0 ))')
        self.SS.connect('((j % rows >=4 and j % rows<=6) and (j // rows == 10 ))')
        #self.SS.connect('j % rows == 5 and j // rows == 10')

    # TODO: Find better way to record spike moments
    def run(self, duration=50*ms, show_PC=False, show_other=False):
        # If show_ = True, then plots voltages

        # StateMonitor will record the voltages values of indices given to record.

        # Records only for a few indices.
        self.MPC = StateMonitor(self.PC, 'v', record=self.my_indices)
        #Records for all indices, used for recording spiking times.
        # TODO: Find better way to record spike moments
        self.MM = StateMonitor(self.PC, 'v', record=True)
        # Records the G inputs of one cell.
        self.MG = StateMonitor(self.G, 'v', record=0)
        # Records the G inputs of one cell.
        self.MS = StateMonitor(self.S, 'v', record=0)
        # Records the inhibitory of one cell.
        self.MINH = StateMonitor(self.INH, 'v', record=0)


        # Records the spikes of Pyramidal cells.
        self.spikemon = SpikeMonitor(self.PC, variables='v', record=True)
        # Records the spikes of one inhibitory cell.
        self.spikemoninh = SpikeMonitor(self.INH,variables='v', record=True)
        self.spikemong = SpikeMonitor(self.G, variables='v', record=True)
        self.spikemons = SpikeMonitor(self.S,variables='v', record=True)


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
        self.INH_all_values = self.spikemoninh.all_values()
        self.G_all_values = self.spikemong.all_values()
        self.S_all_values = self.spikemons.all_values()


        # Following line does not run because, the recorded voltages are empty.
        spiking_times_fun(self, -36.1)

        if show_PC:
            plot_voltages_PC(self)
        if show_other:
            functions.plot_voltages_other_types(self)

