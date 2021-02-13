import numpy as np
#These are functions to generate different connections for different networks. See Section 3.2.1 and Figure 4&5 in Jacob's write up for more details/intuition.


#The following is the only method that should be called. With 'con_type' a string and 'nb_neurons' an int, returns the pre_syn and pos_syn lists. Note that the torus is special, as it requires nb_neurons to be a perfect square (to infer 'cycle1' and 'cycle2').
def generate_connections(con_type, nb_neurons):
    if con_type == 'disconnected':
        return [], []
    elif con_type == 'full':
        return full(nb_neurons)
    elif con_type == 'full_no_loops':
        return full_no_loops(nb_neurons)
    elif con_type == 'simplex':
        return simplex(nb_neurons)
    elif con_type == 'torus':
        #in this case we construct a torus of nb_neurons^2 neurons (each cycle has length nb_neurons)
        return torus(np.sqrt(nb_neurons), np.sqrt(nb_neurons)) 
    elif con_type == 'simplex_torus':
        return simplex_torus(nb_neurons, nb_neurons)
    else:
        print('Connection type not identified.')
        return None


#the following outputs the pre_syn and pos_syn lists to generate a fully connected network on 'nb_neurons' neurons.
def full(nb_neurons):
    pre_syn = [i  for j in range(nb_neurons) for i in range(nb_neurons)]
    pos_syn = [j  for j in range(nb_neurons) for i in range(nb_neurons)]
    return pre_syn, pos_syn

#the following outputs the pre_syn and pos_syn lists to generate a fully connected network with no loops.
def full_no_loops(nb_neurons):
    pre_syn = [j  for j in range(nb_neurons) for i in range(nb_neurons-1)]
    pos_syn = [i  for j in range(nb_neurons) for i in list(range(j))+list(range(j+1, nb_neurons))]
    return pre_syn, pos_syn

#the following outputs the pre_syn and pos_syn lists to generate a fully simplex/clique network on 'nb_neurons' neurons.
def simplex(nb_neurons):#simplex with np_parents vertices
    dim=nb_neurons-1
    pre_syn=[i for i in range(dim) for j in range(dim-i)]
    post_syn=[dim-j for i in range(dim) for j in range(dim-i)]
    return pre_syn, post_syn

#the following outputs the pre_syn and pos_syn lists to generate a fully torus network on 'cylce1'*'cycle2' neurons, where each param corresponds to the length of each cycle, when one thinks of the torus as a cartesian product of two cycles.
def torus(cycle1, cycle2): 
    pre_syn=[y*cycle1+x for y in range(cycle2) for x in range(cycle1)]+[y*cycle1+x for y in range(cycle2) for x in range(cycle1)]
    post_syn=[y*cycle1+((x+1)%cycle1) for y in range(cycle2) for x in range(cycle1)]+[x+cycle1*((y+1)%cycle2) for y in range(cycle2) for x in range(cycle1)]
    return pre_syn, post_syn #graph products?

#Similar to previous, but with diagonals added
def simplex_torus(cycle1, cycle2):
    pre_syn, post_syn = torus(cycle1, cycle2)
    pre_syn+=[(y*cycle1+x) for y in range(cycle2) for x in range(cycle1)]
    post_syn+=[(((y+1)%cycle2)*cycle1+(x+1)%cycle1) for y in range(cycle2) for x in range(cycle1)]
    return pre_syn, post_syn


def parents(nb_p1, nb_p12, nb_p2): # this gives the network connections with two children neurons, nb_p1 parents above first neuron, nb_p2 above second and nb_p12 above both simultaniously
    pre_syn=[x+2 for x in range(nb_p1)]+[x+2 for x in range(nb_p1, nb_p1+nb_p12)]+[x+2 for x in range(nb_p1, nb_p1+nb_p12)]+[x+2 for x in range(nb_p1+nb_p12, nb_p1+nb_p12+nb_p2)] 
    post_syn=[0 for x in range(nb_p1)]+[0 for x in range(nb_p1, nb_p1+nb_p12)]+[1 for x in range(nb_p1, nb_p1+nb_p12)]+[1 for x in range(nb_p1+nb_p12, nb_p1+nb_p12+nb_p2)]
    return pre_syn, post_syn

