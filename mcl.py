'''
Markov Clustering
'''

class MCL:
    def mcl_clustering(self):
        '''
        @ input
            - undirected graph, power parameter e, inflation parameter r
        @ returns
            - resulting cluster matrix
        '''

        # Create the associated matrix

        # Add self loops to each node

        # Normalize the matrix

        # while a steady state is not reached
        # To measure convergence: take the matrix and somehow get the "magnitude"
            # Expand by taking the eth power of the matrix

            # Inflate by taking inflation of the resulting matrix with param r

            # Normalize

            # Pruning


def main():
    # preprocess dataset, returns transition matrix out of it
    # options: weighted/ unweighted
    t_mat = preprocess()

    # clustering

    # visualization


if __name__ == "__main__":
    main()
