from brian2 import *
import functions


class Model:
    def __init__(self, param:dict, model="FairhallModel"):

        self.model = model
        self.reshaped_spikemon = None

        self.number_spiking_cells = None
        self.mean_spike_times = None
        self.variance_spike_times = None

        self.number_spiking_outside_trajectory = None
        self.mean_spikes_outside_trajectory = None
        self.variance_spikes_outside_trajectory = None

        # To check if can use the recorded values.
        self.has_run = False
        self.delay = None

        # Parameters
        self.p = self._complete_parameters(param)
        self._param_sanity_checks(self.p)

        # Spiking times of the cells
        self.PC_spiking_times = None
        self.INH_spiking_times = None

        if self.model == "FairhallModel":
            self._init_pyramidal_neurons_fairhall()
        elif self.model == "ThresholdModel":
            self._init_pyramidal_neurons_threshold()

        self._init_inhibitory_neurons()

        self._init_standard_synapses()

        #Trajectory


        #Noise
        self.N_spiking_times = None
        self._init_noise_neurons()
        self._init_noise_synapses()
        self._init_noise_synapses_2()

        #Random
        self._init_random_input()
        self._init_random_synapses()

        self.neuron_idx = 50
        self.my_indices = functions.some_indices(self.neuron_idx)

        # For plotting purposes
        self.excitatory_matrix = np.zeros([self.p['rows'], self.p['cols']])

    def get_dict_group_neurons(self):
        dict_group_neurons = {}
        dict_group_neurons['PC'] = self.PC
        dict_group_neurons['INH'] = self.INH
        dict_group_neurons['R'] = self.R
        dict_group_neurons['N'] = self.N

        return  dict_group_neurons


    def get_dict_synapses(self):
        dict_synapses = {}
        dict_synapses['PC'] = {}
        dict_synapses['PC']['SPC'] = self.SPC
        dict_synapses['PC']['SPCINH'] = self.SPCINH
        dict_synapses['PC']['SPCN'] = self.SPCN
        dict_synapses['PC']['SPCN2'] = self.SPCN2
        dict_synapses['PC']['SRPC'] = self.SRPC

        dict_synapses['INH'] = {}
        dict_synapses['INH']['SINHPC'] = self.SINHPC

        return dict_synapses


    # Print the parameters. Helps for debugging
    def print_param(self, type_list : str=['INH']):
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

    def set_delay(self, new_delay:float):
        self.delay = new_delay

    def reset_neuron_idx(self, neuron_idx:int):
        self.neuron_idx = neuron_idx

    def reset_neuron_idx(self, list_neurons:list):
        self.my_indices = list_neurons

    # updates parameters of model with p a dictionary of parameters
    def _complete_parameters(self, p : dict):
        """
        :param p: Contains parameters that we want to specify to the model
        :type p: Dictionary
        :return: All parameters necessary to the creation of the model
        :rtype: Dictionary
        """
        genericParam = {
            'rows': 20,
            'cols': 30,
            "thresh_step": 0.2,

            # Excitatory neurons
            "v_init_exc": -60,  # Initial value of potential
            "v_leak_exc": -68,  # Leak potential
            "v_reset_exc": -60,  # Reset potential
            "v_thr_exc": -36,  # Spiking threshold
            'tau_dyn_exc': 50 * ms,  # Leak timescale
            "tau_refr_exc": 8 * ms,  # Refractory period
            "r_max_exc": 20 * Hz,  # Maximum rate
            'rec_weight': 7,  # Recurrent weight
            "lambda_PL_exc": 0.15 * metre,  # ???

            # Inhibitory neurons
            "num_inhib_neurons": 10,
            "v_leak_inh": -10,  # Leak potential
            "v_reset_inh": -80,  # Reset potential
            "v_thr_inh": -50,  # Spiking threshold
            'tau_dyn_inh': 50 * ms,  # Leak timescale
            "tau_refr_inh": 2 * ms,  # Refractory period
            "gi_inh": 1,  # ???
            "inh_weight_pi": 0.02,
            "inh_weight_ip": 0.01,
            #
            # # Noise Neurons
            "num_tonic_noise": 20,  # Number of tonic neurons
            "v_leak_noise": -30,  # Leak potential
            "v_reset_noise": -80,  # Reset potential
            "v_thr_noise": -50,  # Spiking threshold
            'tau_dyn_noise': 5 * ms,  # Leak timescale
            "tau_refr_noise": 2 * ms,  # Refractory period
            "gi_noise": 1,  # ???
            "noise_weight": 0.03,  # Tonic weight

            # Random inputs
            "num_random_neurons": 20,
            "v_leak_random": -30,  # Leak potential
            "v_reset_random": -80,  # Reset potential
            "v_thr_random": -50,  # Spiking threshold
            "input_rate": 50 * Hz,  # Rate of spikes for the
            'R_weight': 3,
            "tau_dyn_random": 5 * ms,  # Leak timescale

            # Synapses
            "w_PC": 0.5,  # Recurrent weight from PC to PC
            "w_PCINH": 0.7,  # Weight PC to INH
            "w_INHPC": 0.5,  # Weight INH to PC
            "w_GPC": 0.9,  # Weight G to PC
            "w_SPC": 0.8  # Weigth S to PC
        }

        pnew = p
        for k, v in genericParam.items():
            if k not in p.keys():
                pnew[k] = v
            else:
                pnew[k] = p[k]

        return pnew

    def _param_sanity_checks(self, p : dict):
        pass

    # If want to modify parameters after creating the class, before running
    def modify_param(self, p : dict):
        if self.has_run:
            raise Warning("The network already ran thus the change of parameters should have no impact")

        for element in p.keys():
            self.p[element] = p[element]

    """
    Initialisation of the cells.
    Created with the brian2 function NeuronGroup.
    It needs an equation.
    """

    # Initialises the pyramidal neurons as a Brian2 NeuronGroup.
    def _init_pyramidal_neurons_fairhall(self):

        # v_leak_exc
        eqs_exc = '''
            dv/dt = (- (v - I)) / tau : 1
            h : 1
            I : 1
            reset : 1
            thresh_step : 1
            x : metre
            y : metre
            tau : second
            '''

        self.PC = NeuronGroup(self.p['rows'] * self.p['cols'], eqs_exc, threshold='v>h', reset='v = - 60',
                              refractory=self.p["tau_refr_exc"], method='euler')
        # initialize the grid positions

        # Variables important for Brian purposes (the equations are written with strings).
        rows = self.p['rows']
        grid_dist = 4 * meter
        self.PC.tau = self.p['tau_dyn_exc']
        # x and y position on the grid of the cells.
        self.PC.x = '(i // rows) * grid_dist'
        self.PC.y = '(i % rows) * grid_dist'
        # Initialises voltage
        self.PC.v = functions.convert_list_to_threshold(self.p['trajectory'][0, :], self.p['v_init_exc'] + 15, self.p['v_init_exc'])
        # self.PC.h = functions.convert_matrix_to_threshold(self.p['connection_matrix_S_fairhall'][0,:],self.p['rows'], self.p['cols'],self.p['v_thr_exc'] - 20, self.p['v_thr_exc'])
        self.PC.h = self.p['v_thr_exc']
        #self.PC.h = functions.convert_list_to_threshold(self.p['trajectory'][0, :], self.p['v_thr_exc'],
         #                                               self.p['v_thr_exc'])
        self.PC.I = functions.convert_list_to_threshold(self.p['trajectory'][0, :], -45, -68)

        #self.PC.reset = functions.convert_list_to_threshold(self.p['trajectory'][0, :], -45, -68)


    # Initialises the pyramidal neurons.
    def _init_pyramidal_neurons_threshold(self):

        # v_leak_exc
        eqs_exc = '''
            dv/dt = (- (v - v_leak_exc)) / tau : 1
            dh/dt = (- (h + 36)) / tau :1
            thresh_step : 1
            x : metre
            y : metre
            tau : second
            '''

        # Here the threshold lowers if the neuron spikes
        self.PC = NeuronGroup(self.p['rows'] * self.p['cols'], eqs_exc, threshold='v>h',
                              reset='v = v_reset_exc; h = h - 2', refractory=self.p["tau_refr_exc"],
                              method='euler')


        self.PC.tau = self.p['tau_dyn_exc']
        self.PC.thresh_step = self.p['thresh_step']

        # initialize the grid positions

        # Variables important for Brian purposes (the equations are written with strings).
        rows = self.p['rows']
        grid_dist = 4 * meter
        # x and y position on the grid of the cells.
        self.PC.x = '(i // rows) * grid_dist'
        self.PC.y = '(i % rows) * grid_dist'

        # Initialises voltage
        self.PC.v = self.p['v_init_exc']

        #Init the threshold
        self.PC.h = -36



    # Initialises the inhibitory neurons as a Brian2 NeuronGroup.
    def _init_noise_neurons(self):

        eqs_inh = '''
        dv/dt = ( - (v - v_leak_noise)) / tau : 1
        tau : second
        '''

        self.N = NeuronGroup(int(self.p['rows'] * self.p['cols'] / 10), eqs_inh, threshold='v>v_thr_noise',
                               reset='v = v_reset_noise', refractory=self.p['tau_refr_noise'], method='euler')
        self.N.tau = self.p['tau_dyn_noise']
        self.N.v = self.p['v_reset_noise']




    # Initialise the inhibitory neurons
    def _init_inhibitory_neurons(self):
        eqs_inh = '''
        dv/dt = ( - (v - v_leak_inh)) / tau : 1
        tau : second
        '''
        # int(self.p['rows'] * self.p['cols'] / 10)
        self.INH = NeuronGroup(int(self.p['rows'] * self.p['cols'] / 10), eqs_inh, threshold='v>v_thr_inh',
                               reset='v = v_reset_inh', refractory=self.p['tau_refr_inh'], method='euler')
        self.INH.tau = self.p['tau_dyn_inh']
        self.INH.v = self.p['v_reset_inh']



    # Initialise the random inputs
    def _init_random_input(self):

        eqs_tonic = '''
        dv/dt = ( - (v - v_leak_random)) / tau : 1
        tau : second
        '''

        self.R = NeuronGroup(self.p['num_random_neurons'], eqs_tonic, threshold='v>v_thr_random',
                             reset='v = v_reset_random',
                             refractory=50 * ms, method='euler')
        self.R.tau = self.p['tau_dyn_random']
        self.R.v = -80



    # Initialises the basic synapses
    def _init_standard_synapses(self):
        # Needed for brian2 equations
        rows = self.p['rows']
        cols = self.p['cols']
        weight = self.p['rec_weight']
        ###########################
        # EXC-EXC synapses
        ###########################
        self.SPC = Synapses(self.PC, self.PC, 'w:1', on_pre='v_post += w')
        self.SPC.connect('(( i // rows - j // rows)**2 + ( i % rows -  j % rows)**2 )< 4')
        # self.SPC.connect()
        self.SPC.w = 'weight*exp(-((x_pre-x_post)**2+(y_pre - y_post)**2)/(30*metre)**2)'

        # If already specified the delay to put, don't modify it.
        if self.delay != None:
            self.SPC.delay = self.delay
        else:
            # self.SPC.delay = '(2 - 2*exp(-(abs(x_pre-x_post)+abs(y_pre - y_post))/(30*metre))) * second'
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


    # Synapses that connect the noise neurons N to cells inside the trajectory
    def _init_noise_synapses(self):

        ##############################################
        # NOISE-(EXC inside the trajectory) synapses
        ##############################################

        self.SPCN = Synapses(self.N, self.PC, 'w:1', on_pre='v_post+=w')

        #For connection
        sources, targets = functions.convert_matrix_to_source_target(self.p['trajectory'])
        self.SPCN.connect(i=sources, j=targets, p=1)

        #Weight of the synapses
        self.SPCN.w = 2 * self.p['noise_weight']

    # Synapses that connect the noise neurons N to cells outside the trajectory
    def _init_noise_synapses_2(self):

        ##############################################
        # NOISE-(EXC outside the trajectory) synapses
        ##############################################

        # Matrix that has 1 outside trajectory and 0 inside
        connection_matrix = np.ones(np.shape(self.p['trajectory'])) + (-1) * self.p['trajectory']

        self.SPCN2 = Synapses(self.N, self.PC, 'w:1', on_pre='v_post+=w')

        #For connection
        sources, targets = functions.convert_matrix_to_source_target(connection_matrix)
        self.SPCN2.connect(i=sources, j=targets, p=0.5)

        #Weight of the synapses
        self.SPCN2.w = self.p['noise_weight']

    # Synapses that connect the random neurons R to pyramidal cells
    def _init_random_synapses(self):
        ###########################
        # RANDOM - EXC synapses
        ###########################
        self.SRPC = Synapses(self.R, self.PC, 'w:1', on_pre='v_post+=w')
        self.SRPC.connect(p=0.01)
        self.SRPC.w = self.p['R_weight']

    # TODO: Find better way to record spike moments
    def run(self, duration=50 * ms, show_PC=False, show_other=False):
        """
        Run the simulation, and record the activity of the desired cells.

        :param duration: Duration in ms of the simulation.
        :type duration: int * ms
        :param show_PC: Ask for plotting the activity of certain cells.
        :type show_PC: bool
        :param show_other: Ask for plotting the activity of different cells.
        :type show_other: bool
        """


        # Records for all indices, used for recording spiking times.
        # TODO: Find better way to record spike moments
        self.MPC = StateMonitor(self.PC, 'v', record=self.my_indices)
        self.MM = StateMonitor(self.PC, 'v', record=True)
        # Threshold
        self.Mthreshold = StateMonitor(self.PC, 'h', record=True)
        # Records the spikes of Pyramidal cells.
        self.spikemon = SpikeMonitor(self.PC, variables='v', record=True)

        # Records the noise inputs N of one cell.
        self.MN = StateMonitor(self.N, 'v', record=0)
        # Records the spikes
        self.spikemonN = SpikeMonitor(self.N, variables='v', record=True)

        self.MR = StateMonitor(self.R, 'v', record=0)
        # Records spkides of random neurons R.
        self.spikemon_random = SpikeMonitor(self.R, variables='v', record=True)

        # Records the inhibitory of one cell.
        self.MINH = StateMonitor(self.INH, 'v', record=0)
        # Records the spikes of one inhibitory cell.
        self.spikemoninh = SpikeMonitor(self.INH, variables='v', record=True)

        # TODO : Find a way to be able to run several times
        if not self.has_run:
            netObjs = {k: v for k, v in vars(self).items() if isinstance(v, BrianObject)}
            self.net = core.network.Network(netObjs)
            self.net.run(duration, namespace=self.p)
        else:
            self.net.run(duration, namespace=self.p)

        # For checking if the network ran
        self.has_run = True

        # Records spiking information about different cells.
        self.PC_all_values = self.spikemon.all_values()
        self.INH_all_values = self.spikemoninh.all_values()
        self.N_all_values = self.spikemonN.all_values()
        self.R_all_values = self.spikemon_random.all_values()

        functions.spiking_times_fun(self)

        if show_PC:
            functions.plot_voltages_PC(self)

        if show_other:
            functions.plot_voltages_other_types(self)
