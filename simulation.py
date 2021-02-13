from brian2 import *
import observationsIO


class Simulation:
    
    """The parameters need to initialize a Simulation are:
    
            - 'nb_neurons': of type int, 
                denoting the total number of neurons present in the network. Each neuron has an index striclty smaller to 'nb_neurons'.
                
            - 'synapse_weight': of type float,
                this denotes the strength of the synapse (see paragraph 2.1 in Jacob's project). In this code, each synapses have equal weights.
                
            - 'time_bin_size' of type int, 
                this denotes the sizer of the time bin used to generate the distribution. Note that this has to divide 'duration'. 
                
            - 'pre_syn': list of int, where members are strictly smaller than 'nb_neurons'.
                These are the pre-synaptic neurons, the i_th entry is the index of the pre-synaptic neuron corresponding to the i-th connection/synapse.
                
            - 'pos_syn': list of int, where members are strictly smaller than 'nb_neurons', and its length is equal to the length of 'pre_syn'.
                These are the post-synaptic neurons, the i_th entry is the index of the post-synaptic neuron corresponding to the i-th connection/synapse.
                
            - 'name': string
                'name' will appear as a prefix in the file_name of the experiment.This is to keep track of experiements.
                
            - 'neurtype', string. optional, default  is 'regular spiking'.
                This is the neuron type considered. See Izhikevich's 'Dynamical systems in neuroscience' for more neuron types. In this code, only two types are available: 'regular spiking' and 'intrinsically bursting', to see raster plots of networks consisting of these neurons please have a look at Jacob's write-up: Figure 6 for 'regular spiking' and figure or Figure 20 for 'intrinsically bursting', or generate your own! :) 
                
            - 'stim': string, either 'on' or 'off'. optional, default is 'off'.
                If 'stim' == 'off', then the simulation will have no external stimulus. If 'stim' is 'on', then neuron 0 (at index 0) will receive a stimulus of 500 between time 250 and 500. Skander's bachelor thesis explores stimulated activity (stim =='on') , Jacob's project explores spontaneous activity (stim=='off').
             
             - 'duration': int, has to be a multiple of time_bin_size.
                 Be carefull wehen reducing this when stim =='on'.  
                 
        The result of a simulation is an object, on which one can run several commands that are written below.
            """
    def __init__(self, nb_neurons, synapse_weight, time_bin_size, pre_syn, pos_syn, name, neurtype ='regular spiking', stim='off', duration=1000):
        # prefs.codegen.target = "numpy"
        defaultclock.dt = 1*ms
        self.nb_neurons = nb_neurons
        self.synapse_weight = synapse_weight
        self.time_bin_size = time_bin_size
        self.duration = duration
        self.nb_bins = duration//time_bin_size 
        self.neurtype=neurtype
        self.name = name
        
        #stim params: These matter only if stim is 'on'
        self.stim_neurons = stim  #should I only treat case where None?
        self.I_MAX = 500 
        self.stim=stim
        
        
        if self.neurtype == 'intrinsically bursting':
            #intrinsically:
            neuron_namespace = {
                'synapse_weight': synapse_weight, #constant input that spike gives post-synaptic neuron
                'a': 0.01, 'b': 5, 'c': -56, 'd': 130, #simple model of choice hyperparam for regular spiking neurons
                'vr': -75, 'vt': -45, 'vpeak': 35, #"_"
                'C': 150, 'k': 1.2, 'tau': 1*ms, #"_"
                #stim param:
                'input_func': TimedArray([0], dt=duration*ms), #initiate variable to be a trivial function
                'I_MAX': self.I_MAX, #maximum stimulus
                'I_on': TimedArray([0, self.I_MAX, self.I_MAX, 0], dt=(duration//4)*ms), #when stimulus is on
                'I_weak': TimedArray([0, self.I_MAX/2, self.I_MAX/2, 0], dt=(duration//4)*ms), #when stim is weak
                'I_off': TimedArray([0], dt=duration*ms) #when stim is off
            }
        elif self.neurtype == 'regular spiking':

            # Regular spiking neurons:
            neuron_namespace = {
                'synapse_weight': synapse_weight, #constant input that spike gives post-synaptic neuron
                'a': 0.03, 'b': -2, 'c': -50, 'd': 100, #simple model of choice hyperparam for regular spiking neurons
                'vr': -60, 'vt': -40, 'vpeak': 35, #"_"
                'C': 100, 'k': 0.7, 'tau': 1*ms, #"_"
                #stim param:
                'input_func': TimedArray([0], dt=duration*ms), #initiate variable to be a trivial function
                'I_MAX': self.I_MAX, #maximum stimulus
                'I_on': TimedArray([0, self.I_MAX, self.I_MAX, 0], dt=(duration//4)*ms), #when stimulus is on
                'I_weak': TimedArray([0, self.I_MAX/2, self.I_MAX/2, 0], dt=(duration//4)*ms), #when stim is weak
                'I_off': TimedArray([0], dt=duration*ms) #when stim is off
            }
        else:
            print('unrecognized neuron type')
        
        #The following code looks like the following comments:
        #model =\
        #'''
        # simple model equation + noise : 1
        # simple model equation : 1
        # stimulus variable : 1
        #initializes a counter : 1
        #'''
        
        if self.stim == 'off':
            model =\
            '''
            dv/dt = (k*(v-vr)*(v-vt) - u + I)/(C*tau) + 5*xi*sqrt(1/tau): 1 
            du/dt = a*(b*(v - vr) - u)/tau : 1
            I = input_func(t) : 1
            nb_spikes_in_bin : 1
            '''
        elif self.stim == 'on':
            model =\
            '''
            dv/dt = (k*(v-vr)*(v-vt) - u + I)/(C*tau) + 5*xi*sqrt(1/tau): 1 
            du/dt = a*(b*(v - vr) - u)/tau : 1
            I = input_func(t)*(i==0) : 1          
            nb_spikes_in_bin : 1
            '''
        else:
            print('Incorrect stim value. pick between "on" or "off".' )
            
        #reseting after a spike has occured:
        
        reset =\
        '''
        nb_spikes_in_bin += 1
        v = c
        u += d
        '''
        peak_threshold = 'v>vpeak' 
        
        #initialize the neurons
        self.neurons = NeuronGroup(self.nb_neurons, model, threshold=peak_threshold, reset=reset,
                         method='euler', namespace=neuron_namespace)
        self.neurons.v = self.neurons.namespace['vr'] 
        self.neurons.u = self.neurons.namespace['b']*self.neurons.v 
        
        
        # Initialize the synapses
        synapse_namespace = {'synapse_weight': synapse_weight} #synapse hyper parameter
        self.S = Synapses(self.neurons, self.neurons, on_pre='v_post += synapse_weight', namespace=synapse_namespace) #creates the synapse type
        
        #Initializes the connections:
        self.S.connect(i=pre_syn,j=pos_syn)
        #Be careful, self.S.connect(i=[], j=[]) makes full conn... Not no connections.
        
        
        # Network
        self.spikemon = SpikeMonitor(self.neurons, record=True)
        self.spikemon.active = False
        
        @network_operation(dt=time_bin_size*ms)
        def update_time_bin(t):
            if t/ms == 0:
                return
            
            if self.stim == 'on':
                obs = tuple([self.neurons.namespace['input_func'](t-defaultclock.dt)] + list(self.neurons.nb_spikes_in_bin))
            else:
                obs = tuple(list(self.neurons.nb_spikes_in_bin)) #If stim =='off' then we only care about the neurons
            bin_index = int((t/ms)/time_bin_size)
            self.observations[bin_index-1].append(obs)
            self.neurons.nb_spikes_in_bin = 0
        
        
        self.network = Network(self.neurons, self.S, update_time_bin, self.spikemon)
        self.network.store()
        
#The following simulates the entire simulation n_monte_carlo times, and writes it down in the folder located in path_to_dir.
    def simulate(self, n_monte_carlo, path_to_dir):
        self.observations = [[] for bin_index in range(self.nb_bins)]
        
        if self.stim == 'on':
            for _ in range(n_monte_carlo):
                self.run_once('I_on')
                
        elif self.stim == 'off':
            for _ in range(n_monte_carlo):
                self.run_once('I_off')
        else:
            print('Incorrect stim value. pick between "on" or "off".')
            
        #The following writes the file. Then file_name depends on almost all parameters of the model (exept connection types, duration...)
        observationsIO.write_observations(
            self.observations, path_to_dir+'{}_nb_neur_{}_sw_{}_tbs_{}_stim_{}'.format(self.name, self.nb_neurons, self.synapse_weight, self.time_bin_size, self.stim))
    
    def run_once(self, input_power):
        self.network.restore()
        self.neurons.namespace['input_func'] = self.neurons.namespace[input_power] #changing the input_power function 
        self.network.run(self.duration*ms + defaultclock.dt)
    
#This runs and plots a simulation. The plot is stored in path_to_file.
    def run_and_plot_example_raster(self, path_to_file):
        self.observations = [[] for bin_index in range(self.nb_bins)]
        self.spikemon.active = True
        
        if self.stim == 'on':
            self.run_once('I_on')
            self.plot_raster('I_on', path_to_file)
        else:
            self.run_once('I_off')
            self.plot_raster('I_off', path_to_file)
        
        self.spikemon.active = False
        
#This runs a simulation and produces the raster plot. The plot is stored in path_to_path_to_dir.  
    def plot_raster(self, input_power, path_to_dir):
        plt_title = '{}_N_{}_sw_{}_tbs_{}_stim_{}'.format(self.name, self.nb_neurons, self.synapse_weight, self.time_bin_size, self.stim)
        fig = figure(figsize=(15, 4))
        title(plt_title)
        plot(self.spikemon.t/ms, self.spikemon.i, '.k')
        xlabel('Time (ms)')
        ylabel('Neuron index')
        yticks(arange(self.nb_neurons))
        for t in arange(0, self.duration + 1, self.time_bin_size):
            axvline(t, ls='--', c='C1', lw=1)
        savefig(path_to_dir+plt_title+'.png')
        plt.close('all')