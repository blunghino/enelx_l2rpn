import numpy as np 
import networkx as nx


class ObservationSpaceGraph(object):

    def __init__(self, observation_space):
        self.observation_space = observation_space
        self.observation_dict = dict()
        self.nodes = list()
        self.edges = list()

    def get_graph(self, obs_array):
        self.observation_dict = vars(self.observation_space.array_to_observation(obs_array))
        self.__get_nodes()
        self.__get_edges()
        return({'obs_G': self.__build_graph(),
                'obs_DG': self.__build_directed_graph()})        

    def __build_graph(self):
        # Build the undirected graph
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)
        return G

    def __build_directed_graph(self):
        # Build the undirected graph
        DG = nx.DiGraph()
        DG.add_nodes_from(self.nodes)
        DG.add_weighted_edges_from(self.edges)
        return DG

    def __get_nodes(self):
        self.nodes = self.observation_dict.get('substations_ids').tolist()

    def __get_edges(self):

        # Get the edges
        self.edges = list(zip(self.observation_dict.get('lines_or_substations_ids'), self.observation_dict.get('lines_ex_substations_ids')))

        # Get the line weights
        ampere_flows = [flow for flow in self.observation_dict.get('ampere_flows')]

        # Get the line limits
        thermal_limits = [limit for limit in self.observation_dict.get('thermal_limits')]

        # Build the edges with properties
        for i in range(0, len(self.edges)):
            edge_properties = {
                'weight': ampere_flows[i],  # weight is a special property that is used in graph theory
                'limit': thermal_limits[i]
            }
            self.edges[i] = tuple([self.edges[i][0], self.edges[i][1], edge_properties])

    def get_state_with_graph_features(self, obs_array):
        """
        Feature engineering for observation space

        Parameters
        ----------
        obs_array : numpy.ndarray
            Observation vector to use to generate graph based features

        Returns
        -------
        np.ndarray
            state space vector with graph engineered features
        """
        graphs = self.get_graph(obs_array)
        # This was originally designed for ranking web pages
        # The result is a ranking of the substations in the grid taking into account their connectedness & weight (flows)
        # We might want to toggle the substations with the highest or lowest values
        pagerank_ = nx.pagerank(graphs['obs_G'])

        # The sum of the fraction of all-pairs shortest paths that passes through node in key
        # This is a potential measure of how critical a substation is for connecting the grid
        betweenness_centrality_ = nx.betweenness_centrality(graphs['obs_G'])

        # Measure of how central a substation is in the grid
        degree_centrality_ = nx.degree_centrality(graphs['obs_G'])

        # Measure of how central a substation is as a producer (uses a directed graph)
        out_degree_centrality_ = nx.out_degree_centrality(graphs['obs_DG'])

        # Measure of how central a substation is as a consumer (uses a directed graph)
        in_degree_centrality_ = nx.in_degree_centrality(graphs['obs_DG'])

        pagerank = []
        betweenness_centrality = []
        degree_centrality = []
        out_degree_centrality = []
        in_degree_centrality = []

        for k in sorted(pagerank_.keys()):
            pagerank.append(pagerank_[k])
            betweenness_centrality.append(betweenness_centrality_[k])
            degree_centrality.append(degree_centrality_[k])
            out_degree_centrality.append(out_degree_centrality_[k])
            in_degree_centrality.append(in_degree_centrality_[k])

        return np.hstack((
            obs_array,
            pagerank,
            betweenness_centrality,
            degree_centrality,
            out_degree_centrality,
            in_degree_centrality
        ))
