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
        #print ("normalized: ")
        # print (self.t_mat)
        #print ()

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
            # print("diff: ", diff)
            if diff < 0.0001:
                is_steady = True
            #print ()

        #print (self.t_mat)
        # assessment 1 : modularity
        modularity = self.get_modularity(t_mat_cp)
        print("modularity:", modularity)
        conductance = self.get_conductance(t_mat_cp)
        print("conductance: ", conductance)
        coverage = self.get_coverage(t_mat_cp)
        print("coverage: ", coverage)

        self.visualize(t_mat_cp)

    def normalize(self):
        for i in range(len(self.t_mat)):
            col = self.t_mat[:, i]
            col_sum = col.sum()
            col = col/col_sum
            self.t_mat[:, i] = col

    def get_modularity(self, mat):
        '''
        Assessment 1: Modularity
        '''
        modularity = 0
        E = np.count_nonzero(mat)
        edges = set()
        for i in range(len(self.t_mat)):
            row = self.t_mat[i,:]
            nonzero_indices = np.where(row!=0)[0]
            if len(nonzero_indices) > 0:
                inter_cluster_edges = itertools.combinations(nonzero_indices, 2)
                cur_edges = []
                unique_edges = 0
                # Finding inter cluster edges
                for edge in inter_cluster_edges:
                    if mat[edge[0],edge[1]] == 1:
                        if edge not in edges:
                            cur_edges.append(edge)
                            edges.add(edge)
                            unique_edges += 1
                ekk = 2 * len(cur_edges)
                ak = 0
                if unique_edges != 0:
                    for j in range(len(mat)):
                        if row[j] != 0:
                            ak_list = np.where(mat[j,:]!=0)[0]
                            ak = ak + len(ak_list)
                print ("ekk: ", ekk, " ak: ", ak, " E: ", E)
                modularity = modularity + ((ekk/E)-(ak/E)**2)
        return modularity

    def get_conductance(self, mat):
        '''
        Assessment 2: Conductance
        '''
        conductance = 0
        E = np.count_nonzero(mat)/2
        k = 0
        edges = set()
        for i in range(len(self.t_mat)):
            row = self.t_mat[i,:]
            nonzero_indices = np.where(row!=0)[0]
            if len(nonzero_indices) > 0:
                inter_cluster_edges = itertools.combinations(nonzero_indices, 2)
                cur_edges = []
                unique_edges = 0
                # Finding inter cluster edges
                for edge in inter_cluster_edges:
                    if mat[edge[0],edge[1]] == 1:
                        if edge not in edges:
                            cur_edges.append(edge)
                            edges.add(edge)
                            unique_edges += 1
                intra_edges = len(cur_edges)
                aj = 0

                # Calculating aj, any edges within or that connect to the cluster
                for j in range(len(mat)):
                    if row[j] != 0:
                        aj_list = np.where(mat[j,:]!=0)[0]
                        aj = aj + len(aj_list)
                Aij = aj - (2 * intra_edges)
                Ask = intra_edges + Aij
                Askc = E - intra_edges


                # only compute conductance if cluster is unique
                if unique_edges != 0:
                    conductance = conductance + (Aij/min(Ask, Askc))
                    print ("aj: ", aj, "Aij: ", Aij, " Ask: ", Ask, " Askc: ", Askc)
                    k += 1
        conductance = 1 - (conductance / k)
        print("K: ", k)
        return conductance

    def get_coverage(self, mat):
        '''
        Assessment 3: Coverage
        '''
        coverage = 0
        E = np.count_nonzero(mat)/2
        intra_edges = 0
        edges = set()
        for i in range(len(self.t_mat)):
            row = self.t_mat[i,:]
            nonzero_indices = np.where(row!=0)[0]
            if len(nonzero_indices) > 0:
                inter_cluster_edges = itertools.combinations(nonzero_indices, 2)
                # Finding inter cluster edges
                for edge in inter_cluster_edges:
                    if mat[edge[0],edge[1]] == 1:
                        edges.add(edge)
        intra_edges = len(edges)
        coverage = intra_edges/E
        print ("E: ", E, "cluster edges:", intra_edges)
        return coverage

    def visualize(self, mat):
        G = nx.Graph()
        for i in range(len(mat)):
            G.add_node(i)
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

        #print(cluster_edges)

        graph_pos = nx.fruchterman_reingold_layout(G)

        nx.draw_networkx_nodes(G, graph_pos, node_size=300, node_color='blue', alpha = 0.3)
        nx.draw_networkx_edges(G, graph_pos)

        base_color = 0
        color = ["r", "g", "b", "m", "y", "pink", "purple", "orange", "dodgerblue", "chartreuse", "dimgrey", "khaki", "coral", "maroon"]
        for edges in cluster_edges:
            #print ("edges: ", edges)
            nx.draw_networkx_edges(G, graph_pos, edgelist=edges, width=8, alpha=0.5, edge_color=color[base_color % len(color)])
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
            if  len(new_data) == 2:
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
    # filename = "animal_data/dolphins/out.dolphins"
    filename = "animal_data/moreno_zebra/out.moreno_zebra_zebra"
    # filename = 'test_data.txt'
    t_mat = preprocess(filename)

    # clustering
    e = 3
    r = 5
    mcl = MCL(t_mat, e, r)
    mcl.mcl_clustering()
    # visualization


if __name__ == "__main__":
    main()
