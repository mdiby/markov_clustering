Markov Clustering
===

Eunjin Cho, Rachel Cheung, Tegan Wilson
Data Mining, Spring 2017

1. How to Run Our Code
- `python3 mcl.py <FILE_PATH> <IS_WEIGHTED> <E> <R>`
  - e.g. `python3 mcl.py animal_data/dolphins/out.dolphins False`
  - <FILE_PATH>: the path to the file,
                 file needs to be in form of (node1 node2) for unweighted files
                 and (node1 node2 weight) for weighted files
  - <IS_WEIGHTED>: if the file is weighted, True
                   if the file is "un"weighted, False
  - <E>: the expansion parameter, must be a positive integer larger than 1
  - <R>: the inflation parameter, must be a positive real number

- To run our code, you need the packages numpy, scipy, networkx, and matplotlib
installed.

2. Code Result
The code prints out three assessment values (modularity, conductance, coverage).
It also visualizes the clusterings of given data, such that the edges of a given cluster
are all the same color.

3. Known Bugs
- Our assessment values only apply to unweighted graphs. Sometimes the assessment values
may not be computable on weighted graphs.
- For certain e or r, assessment values may not be computable. This is not a bug,
just an inadequacy of the measurements given certain clusters (i.e. dividing by 0).
- If there are more than 15 clusters, some clusters will have the same color
edges despite being different clusters.

4. Included Datasets
We included two unweighted datasets (dolphins and zebras). You can call the datasets as following:
animal_data/dolphins/out.dolphins
animal_data/moreno_zebra/out.moreno_zebra_zebra

We included one weighted dataset (cattle). You can call the dataset as following:
animal_data/moreno_cattle/out.moreno_cattle_cattle
