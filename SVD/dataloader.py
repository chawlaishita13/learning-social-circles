import os
import numpy as np
import networkx as nx

class SVDDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.graph, self.user_mapping = self.load_data()
        self.adj_matrix = self.build_adjacency_matrix()
    
    def load_data(self):
        """Load .circles files and construct the graph."""
        graph = nx.Graph()
        user_mapping = {}
        user_counter = 0
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".circles"):
                file_path = os.path.join(self.data_dir, filename)
                ego_user = filename[:-8]  # Extract ego user ID
                
                if ego_user not in user_mapping:
                    user_mapping[ego_user] = user_counter
                    user_counter += 1
                
                graph.add_node(ego_user)
                with open(file_path, "r") as f:
                    for line in f:
                        parts = line.strip().split(":")
                        if len(parts) < 2:
                            continue
                        _, members = parts
                        members = members.split()
                        
                        for member in members:
                            if member not in user_mapping:
                                user_mapping[member] = user_counter
                                user_counter += 1
                            graph.add_edge(ego_user, member)
                            for other_member in members:
                                if member != other_member:
                                    graph.add_edge(member, other_member)
        
        return graph, user_mapping
    
    def build_adjacency_matrix(self):
        """Build adjacency matrix for SVD."""
        num_users = len(self.user_mapping)
        adj_matrix = np.zeros((num_users, num_users))
        
        for user1, user2 in self.graph.edges():
            i, j = self.user_mapping[user1], self.user_mapping[user2]
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1  # Undirected graph
        
        return adj_matrix
