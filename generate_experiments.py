import gen_connections
import simulation
import numpy as np
import new_it_analyzer

#These generate Jacob's experiments, they summarize what is done in the ParentConnComparer and CommonParents notebooks. 


"""The following corresponds to Section 4.1 in Jacob's. This essentially loops through all the hyperparameter lists and generates one experiement for each. The parameters are the following:
    - 'parent_con_types': string , has to be well defined as an argument from gen_connection.
    - 'nb_parents_list': list of ints. 
        Each member of the list will be considered individually as a number of parents for the experiment. 
    - 'time_bin_sizes' = list of ints, has to divide duration (=1000, see simulation). 
        Each member of this list will be considered individually as a time bin size.
    - 'synapse_weights': list of ints,
        Each member of this list will be considered individually as a synapse weight.
    - 'neuron_type': string, has to be able to enter as argument in simulation class.
    - 'n_monte_carlo': int, 
        Number of times each experiments are repeated.
"""

#See notebook for examples.
def parents_one_child(parent_con_type='disconnected', nb_parents_list=np.arange(1,40, 2), time_bin_sizes=[50], synapse_weights=[10], neuron_type='regular_spiking',n_monte_carlo=500):
    
    for tbs in time_bin_sizes:
        for synapse_weight in synapse_weights:
            for nb_parents in nb_parents_list:
                PATH_TO_DIR = 'observations/parents_one_child/'+parent_con_type+'/'#make sure that you have this path, and that in the folder there is a raster folder!
                nb_neurons=nb_parents+1
            
                #for parent child conn
                pre_syn=[i for i in range(1, nb_parents+1)] 
                pos_syn=[0 for x in range(1, nb_parents+1)]
            
                #inter-parent-connection:
                parent_pre_syn, parent_pos_syn = gen_connections.generate_connections(parent_con_type, nb_parents)
                pre_syn+=[x+1 for x in parent_pre_syn]
                pos_syn+=[x+1 for x in parent_pos_syn]

                ex=simulation.Simulation(nb_neurons, synapse_weight, tbs, pre_syn, pos_syn, 'parents_{}_child_1'.format(nb_parents), stim='off', neurtype = neuron_type)
                ex.simulate(n_monte_carlo, PATH_TO_DIR)
                ex.run_and_plot_example_raster(PATH_TO_DIR+'raster/') #make sure you have a raster folder in the directiory
    return None

"""Similar to above, but this corresponds to Section 4.2 and 4.3. Now we have two children, and therefore a new parameter: 
    - children_con_type_list: list of string,
        Each string is considered individually as the inter-children connection type"""

def parents_two_children(parent_con_type='disconnected', nb_parents_list=np.arange(1,20, 2), children_con_type_list=['disconnected'], time_bin_sizes=[50], synapse_weights=[10], neuron_type='regular_spiking', n_monte_carlo=500):
    if parent_con_type == 'torus':
        print('this does no work for torus connections...')
        return None
    for tbs in time_bin_sizes:
        for synapse_weight in synapse_weights:
            for nb_parents in nb_parents_list:
                for children_con_type in children_con_type_list:
                    PATH_TO_DIR = 'observations/parents_two_children/' + parent_con_type+'_parents/' + children_con_type + '_children/' #Make sure that this path exists, and that in the folder there is a raster folder!
                    nb_neurons=nb_parents+2

                    #for parent child conn
                    pre_syn=[i for i in range(2, nb_parents+2)]+[i for i in range(2, nb_parents+2)] 
                    pos_syn=[0 for x in range(2, nb_parents+2)]+[1 for x in range(2, nb_parents+2)] 

                    #inter-parent-connection:
                    parent_pre_syn, parent_pos_syn = gen_connections.generate_connections(parent_con_type, nb_parents)
                    pre_syn+=[x+2 for x in parent_pre_syn]
                    pos_syn+=[x+2 for x in parent_pos_syn]

                    ex=simulation.Simulation(nb_neurons, synapse_weight, tbs, pre_syn, pos_syn, 'parents_{}_child_2'.format(nb_parents), stim='off', neurtype = neuron_type)
                    ex.simulate(n_monte_carlo, PATH_TO_DIR)
                    ex.run_and_plot_example_raster(PATH_TO_DIR+'raster/') #make sure you have a raster folder in the directiory
    return None
        

    "This corresponds to Section 4.4 is Jacob's write-up"
def three_neur_motifs(time_bin_sizes=[50], synapse_weights=[10], neuron_params=['regular_spiking'],n_monte_carlo=500):
    
    possible_con_3_neur=[([0],[1]), ([0, 1], [1, 0]), ([0, 1],[1, 2]), ([0, 2], [1, 1]), ([1, 1], [0, 2]), ([0, 1, 1],[1, 0, 2]), ([0, 1, 2], [1, 0, 1]), ([0, 0, 1], [1,2, 2]), ([0, 1, 2], [1, 2, 0]), ([0, 1, 1, 2], [1, 0, 2, 1]), ([0, 1, 1, 2], [1, 0, 2, 0]), ([0, 0, 1, 1], [1, 2, 0, 2]), ([0, 1, 2,2], [1, 0, 0, 1]), ([0, 1, 1, 2, 2], [1, 0, 2, 0, 1]), ([0,0,1,1,2,2], [1,2,0,2,0,1])]

    for tbs in time_bin_sizes:
        for synpase_weight in synapse_weights:
            for graph_type in range(len(possible_con_3_neur)):
                PATH_TO_DIR = f'observations/three_neuron_motifs/'
                pre_syn=[]
                post_syn=[]
                pre_syn=possible_con_3_neur[graph_type][0]
                post_syn=possible_con_3_neur[graph_type][1]
                ex=simulation.Simulation(3, synapse_weight, tbs, pre_syn, post_syn, 'graph_type_{}'.format(graph_type))
                ex.run_and_plot_example_raster(PATH_TO_DIR+'raster/')
                ex.simulate(n_monte_carlo, PATH_TO_DIR)
    return None



#These were used to plot the Figures in the above sections:

def generate_children_mut_inf(connection_type, xaxis, sw, tbs, children_connection_type='disconnected'):
    PATH_TO_DIR = 'observations/parent_two_children/'+connection_type+'_parents/' + children_connection_type + '_children/'
    analyzs=[new_it_analyzer.IT_analyzer(PATH_TO_DIR+'parents_{}_child_2_nb_neur_{}_sw_{}_tbs_{}_stim_off'.format(i, i+2, sw, tbs)) for i in xaxis]
    children_mutual_inf=[np.mean([dit.shannon.mutual_information(d, [0], [1], rv_mode='indices') for d in analyz.dists]) for analyz in analyzs]
    return children_mutual_inf

#same but conditioned on parents
def generate_children_cond_mut_inf(connection_type, xaxis, sw, tbs, children_connection_type='disconnected'):
    PATH_TO_DIR = 'observations/parent_two_children/'+connection_type+'_parents/' + children_connection_type + '_children/'
    analyzs=[new_it_analyzer.IT_analyzer(PATH_TO_DIR+'parents_{}_child_2_nb_neur_{}_sw_{}_tbs_{}_stim_off'.format(i, i+2, sw, tbs)) for i in xaxis]
    children_mutual_inf=[np.mean([dit.shannon.conditional_entropy(d, [0], range(2,analyz.nb_neurons), rv_mode='indices')-dit.shannon.conditional_entropy(d, [0], range(1, analyz.nb_neurons), rv_mode='indices') for d in analyz.dists]) for analyz in analyzs]
    return children_mutual_inf

def generate_child_entropy_siblings(connection_type, xaxis, sw, tbs, children_connection_type=None):
    PATH_TO_DIR = 'observations/parent_two_children/'+connection_type+'_parents/' + children_connection_type + '_children/'
    analyzs=[new_it_analyzer.IT_analyzer(PATH_TO_DIR+'parents_{}_child_2_nb_neur_{}_sw_{}_tbs_{}_stim_off'.format(i, i+2, sw, tbs)) for i in xaxis]
    child_entropy=[np.mean([dit.shannon.entropy(d, [0], rv_mode='indices') for d in analyz.dists]) for analyz in analyzs]
    return child_entropy


#A nice way to organize this would be to do an experiment_analysis class, combining Skander'S it_analyzer class to this.