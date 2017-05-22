'''
Markov Clustering
'''
import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
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

        t_mat_cp = np.copy(self.t_mat)

        # Add self loops to each node
        for i in range(len(self.t_mat)):
            self.t_mat[i, i] = 1

        # Normalize the matrix
        self.normalize()
        print ("normalized: ")
        print (self.t_mat)
        print ()

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
        self.visualize(t_mat_cp)

    def normalize(self):
        for i in range(len(self.t_mat)):
            col = self.t_mat[:, i]
            col_sum = col.sum()
            col = col/col_sum
            self.t_mat[:, i] = col

    def visualize(self, mat):
        G = nx.Graph()
        for i in range(len(mat)):
            row = mat[i, :]
            nonzero_indices = np.where(row!=0)[0]
            for ind in nonzero_indices:
                G.add_edge(i, ind)

        cluster_nodes = []
        cluster_edges = []
        for i in range(len(self.t_mat)):
            row = self.t_mat[i,:]
            nonzero_indices = np.where(row!=0)[0]
            cur_edges = []
            if len(nonzero_indices) > 0:
                cluster_nodes.append(nonzero_indices)
                edge_perm = itertools.combinations(nonzero_indices, 2)
                for edge in edge_perm:
                    if mat[edge[0],edge[1]] == 1:
                        cur_edges.append(edge)
            if len(cur_edges) > 0:
                cluster_edges.append(cur_edges)

        print(cluster_edges)

        graph_pos = nx.spring_layout(G)

        nx.draw_networkx_nodes(G, graph_pos, node_size=1000, node_color='blue', alpha = 0.3)
        nx.draw_networkx_edges(G, graph_pos)

        base_color = 0
        color = ["r", "g", "b", "m", "y", "pink", "purple", "orange", "dodgerblue", "chartruesse", "dimgrey"]
        for edges in cluster_edges:
            print ("edges: ", edges)
            nx.draw_networkx_edges(G, graph_pos, edgelist=edges, width=8, alpha=0.5, edge_color=color[base_color])
            base_color += 1
        nx.draw_networkx_labels(G, graph_pos, font_size=12, font_family='sans-serif')
        plt.show()

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
    #filename = "animal_data/dolphins/out.dolphins"
    filename = 'test_data.txt'
    t_mat = preprocess(filename)

    # clustering
    e = 2
    r = 2
    mcl = MCL(t_mat, e, r)
    mcl.mcl_clustering()
    # visualization


if __name__ == "__main__":
    main()
