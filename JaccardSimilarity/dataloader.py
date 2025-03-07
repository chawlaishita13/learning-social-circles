import os
import networkx as nx

class JaccardDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.graph = self.load_data()
    
    def load_data(self):
        """Load .circles files and construct the graph."""
        graph = nx.Graph()
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".circles"):
                file_path = os.path.join(self.data_dir, filename)
                ego_user = filename[:-8]  # Extract ego user ID
                graph.add_node(ego_user)
                
                with open(file_path, "r") as f:
                    for line in f:
                        parts = line.strip().split(":")
                        if len(parts) < 2:
                            continue
                        _, members = parts
                        members = members.split()
                        
                        # Add edges
                        for member in members:
                            graph.add_edge(ego_user, member)
                            for other_member in members:
                                if member != other_member:
                                    graph.add_edge(member, other_member)
        
        return graph
