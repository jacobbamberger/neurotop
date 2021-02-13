import numpy as np
import dit #this is the library used for the information theory: https://dit.readthedocs.io/en/latest/generalinfo.html
from observationsIO import read_observations

class IT_analyzer:
    def __init__(self, path_to_file, stim='off'):
        self.bins_of_observations = read_observations(path_to_file)#shape (number of bins) x (number of experiments) x (number of random variables per bin). This is equal to (nb_bins, n_monte, stim+nbneur) with stim=0 if 'off', 1 o.w. . 
        self.nb_time_bins = len(self.bins_of_observations)
        self.stim = stim
        if stim == 'off' :
            self.nb_neurons = len(self.bins_of_observations[0][0])
        else:
            self.nb_neurons = len(self.bins_of_observations[0][0])-1
            
        self.dists = self.generate_distributions() 
        

    def generate_distributions(self):
        #Outputs a list of nb_time_bins=DUR/tbs distributions which are dictionaries of probabilities with keys being the vectors encoding firing info. Note that tbs has to divide DUR.
        dists = []
        for bin_index, observations in enumerate(self.bins_of_observations):
            if self.stim == 'on': #if I is nontrivial
                dists.append(
                    self.generate_prob_distribution(
                        observations, #observations has shape (DUR/tbs, 1+nbneur)
                        var_names= ['I'] + ['Neuron_{}_bin_{}_Spk_count'.format(
                            neuron_index, bin_index) for neuron_index in range(self.nb_neurons)]))
            else:
                dists.append(
                    self.generate_prob_distribution(
                        observations, #observations has shape (DUR/tbs, nbneur)
                        var_names= ['Neuron_{}_bin_{}_Spk_count'.format(
                            neuron_index, bin_index) for neuron_index in range(self.nb_neurons)]))
        return dists
    
    def generate_prob_distribution(self, observations,  var_names=None):
        #Inputs a list of shape (n_monte, stim+nb_neur) and outputs a ditribution by averaging on the n_monte axis 
        pmf = {}

        for obs in observations:#obs is a nbneur+nbstim sized vector
            pmf[obs] = pmf.get(obs, 0) + 1 #counts nb times this vector appeared

        for obs in pmf:
            pmf[obs] /= len(observations) #normalize by nb of vectors in this time bin

        d = dit.Distribution(pmf)

        if var_names is not None:
            d.set_rv_names(var_names)#sets names to the variables.
        
        return d
    

    #The following is used in Skander's Bachelor thesis.
    
    def compute_stimulus_encodings(self):
        return [[self.NMI(d, [0], [neuron_idx]) for d in self.dists] for neuron_idx in range(1, self.nb_neurons+1)]
    
    
    def NMI(self, d, vars1, vars2):#normalized mutual information
        MI = dit.shannon.mutual_information(d, vars1, vars2, rv_mode='indices')
        Hvars1 = dit.shannon.entropy(d, vars1, rv_mode='indices')
        Hvars2 = dit.shannon.entropy(d, vars2, rv_mode='indices')
        return 2*MI/(Hvars1 + Hvars2)

    def NCMI(self, d, vars1, vars2, cond):#normalized conditional mutual information
        CMI = dit.shannon.conditional_entropy(d, vars1, cond, rv_mode='indices') - dit.shannon.conditional_entropy(d, vars1, vars2+cond, rv_mode='indices')
        Hvars1 = dit.shannon.entropy(d, vars1, rv_mode='indices')
        Hvars2 = dit.shannon.entropy(d, vars2, rv_mode='indices')
        return 2*CMI/(Hvars1 + Hvars2)
    
    def PID(self, d, X1, X2, Y):#partial information decomposition
        """
        Return [I, S, R, U1, U2]
        """
        HX = dit.shannon.entropy(d, X1 + X2, rv_mode='indices')
        HY = dit.shannon.entropy(d, Y, rv_mode='indices')
    
        I1 = dit.shannon.mutual_information(d, X1, Y, rv_mode='indices')
        I2 = dit.shannon.mutual_information(d, X2, Y, rv_mode='indices')
        
        MI = dit.shannon.mutual_information(d, X1 + X2, Y, rv_mode='indices')
        
        R = self.Imin(d, [X1, X2], Y)
        
        U1 = I1 - R
        U2 = I2 - R
        S = MI - R - U1 - U2
        
        return 2*np.array([MI, S, R, U1, U2])/(HX + HY)
    
    def Imin(self, d, Xs, Y):
        """ Xs == [X1, X2] """
        
        def PMI(d, X, Y, y):
            """
            Pointwise Mutual Information for a value y of Y
            """
            PY, PX_c_Y = d.condition_on(Y, rvs=X, rv_mode='indices')
            Py = PY[y]
            PX_c_y = PX_c_Y[PY.outcomes.index(y)]
            PX, PY_c_X = d.condition_on(X, rvs=Y, rv_mode='indices')
            Py_c_X = {x: PY_c_x[y] for x, PY_c_x in zip(PX.outcomes, PY_c_X)}

            return np.nansum([PX_c_y[x] * np.log2(Py_c_x / Py) for x, Py_c_x in Py_c_X.items()])
    
        PY = d.marginal(Y, rv_mode='indices')
        return sum(PY[y] * min(PMI(d, X, Y, y) for X in Xs) for y in PY.outcomes)
    
    
class Simplex_IT_analyzer(IT_analyzer): #used in Skander's Bachelor thesis.
    def __init__(self, path_to_file, stim='on'):
        super().__init__(path_to_file, stim='on')
        
    def compute_neuron_mutual_informations(self):
        source_to_postsource = [self.NMI(d, [1], [2]) for d in self.dists]
        source_to_sink = [self.NMI(d, [1], [self.nb_neurons]) for d in self.dists]
        beforesink_to_sink = [self.NMI(d, [self.nb_neurons-1], [self.nb_neurons]) for d in self.dists]
        return [source_to_postsource, source_to_sink, beforesink_to_sink]
    
    def compute_conditional_mutual_informations(self):
        source_to_postsource = [self.NCMI(d, [1], [2], [0]) for d in self.dists]
        source_to_sink = [self.NCMI(d, [1], [self.nb_neurons], [0]) for d in self.dists]
        beforesink_to_sink = [self.NCMI(d, [self.nb_neurons-1], [self.nb_neurons], [0]) for d in self.dists]
        return [source_to_postsource, source_to_sink, beforesink_to_sink]
    
    def compute_partial_information_decompositions(self):
        source_to_postsource = np.column_stack([self.PID(d, [0] ,[1], [2]) for d in self.dists])
        source_to_sink = np.column_stack([self.PID(d,[0], [1], [self.nb_neurons]) for d in self.dists])
        beforesink_to_sink = np.column_stack([self.PID(d, [0], [self.nb_neurons-1], [self.nb_neurons]) for d in self.dists])
        return [source_to_postsource, source_to_sink, beforesink_to_sink]
    
    
class Experiment_IT_analyzer:
    
    def __inti__(self, path_to_dir, xaxis, tbs, sw):
        self.nb_children=2 #this is adapted to the experiment to two children
        self.path_to_dir= 'observations/parent_two_children/'+connection_type+'_parents/' + children_connection_type + '_children/'
        self.analyzs= [[IT_analyzer(PATH_TO_DIR+'parents_{}_child_2_nb_neur_{}_sw_{}_tbs_{}_stim_off'.format(i, i+2, sw, tbs)) for i in xaxis]]