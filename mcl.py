'''
Markov Clustering
'''
import scipy
import numpy as np
np.set_printoptions(threshold=np.inf)

class MCL:
    def __init__(self, t_mat, e, r):
        self.t_mat = t_mat
        self.e = e
        self.r = r


    def mcl_clustering(self):
        '''
        @ input
            - undirected graph, power parameter e, inflation parameter r
        @ returns
            - resulting cluster matrix
        '''

        # Add self loops to each node
        for i in range(len(self.t_mat)):
            self.t_mat[i, i] = 1

        # Normalize the matrix
        self.normalize()
        print ("normalized: ")
        print (self.t_mat)
        print ()
        t_mat_cp = np.copy(self.t_mat)

        is_steady = False
        # while a steady state is not reached
        while not is_steady:
            previous = np.copy(self.t_mat)
        # To measure convergence: take the matrix and somehow get the "magnitude"
            # Expand by taking the eth power of the matrix
            self.t_mat = np.linalg.matrix_power(self.t_mat, self.e)
            #print ("after e: ")
            #print (self.t_mat)

            # Inflate by taking inflation of the resulting matrix with param r
            self.t_mat = np.power(self.t_mat, self.r)
            #print ("after r: ")
            #print (self.t_mat)

            # Normalize
            self.normalize()
            #print ("after normalization: ")
            #print (self.t_mat)

            # Pruning
            self.t_mat[self.t_mat < 0.01] = 0
            #print ("after pruning: ")
            #print(self.t_mat)

            # Test for steady state
            diff_mat = np.subtract(previous, self.t_mat)
            diff_mat = np.absolute(diff_mat)
            diff = diff_mat.sum()
            #print(diff_mat)
            print("diff: ", diff)
            if diff < 0.0001:
                is_steady = True
            #print ()

        print (self.t_mat)

    def normalize(self):
        for i in range(len(self.t_mat)):
            col = self.t_mat[:, i]
            col_sum = col.sum()
            col = col/col_sum
            self.t_mat[:, i] = col

def preprocess(filename):
    '''
    Parse the data from filename
    @ returns
        - data_matrix: numpy transition matrix of graphs for given file
    '''
    # {key: node, value: set of nodes that are connected to key}
    data_dict = dict()
    with open(filename, 'r') as f:
        for line in f:
            new_data = line.split()
            if (not len(new_data) == 3):
                node1 = int(new_data[0])
                node2 = int(new_data[1])
                if (node1 in data_dict):
                    data_dict[node1].add(node2)
                elif (node1 not in data_dict):
                    data_dict[node1] = set()
                    data_dict[node1].add(node2)
                if (node2 in data_dict):
                    data_dict[node2].add(node1)
                else:
                    data_dict[node2] = set()
                    data_dict[node2].add(node1)

    # initialize matrix (numpy array of array)
    data_matrix = np.zeros((len(data_dict), len(data_dict)))

    # fill in the matrix
    for node, connect_set in data_dict.items():
        for connode in connect_set:
            data_matrix[node-1, connode-1] = 1

    return data_matrix


def main():
    # preprocess dataset, returns transition matrix out of it
    # options: weighted/ unweighted
    filename = "animal_data/dolphins/out.dolphins"
    #filename = 'test_data.txt'
    t_mat = preprocess(filename)

    # clustering
    e = 2
    r = 2
    mcl = MCL(t_mat, e, r)
    mcl.mcl_clustering()
    # visualization


if __name__ == "__main__":
    main()
