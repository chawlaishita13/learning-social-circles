class JaccardSimilarityModel:
    def __init__(self, graph):
        self.graph = graph
    
    def compute_jaccard_similarity(self, node1, node2):
        """Compute Jaccard Similarity between two nodes."""
        neighbors1 = set(self.graph.neighbors(node1)) if node1 in self.graph else set()
        neighbors2 = set(self.graph.neighbors(node2)) if node2 in self.graph else set()
        
        intersection_size = len(neighbors1 & neighbors2)
        union_size = len(neighbors1 | neighbors2)
        
        return intersection_size / union_size if union_size > 0 else 0.0
    
    def predict_links(self, test_edges, threshold=0.01): 
        """Given an edge, determine if it is a true edge (they're actually friends), or not

        Args:
            test_edges (list): List of edges (user1, user2) pairs
            threshold (float, optional): Threshold for jaccard similarity. Defaults to 0.01.

        Returns:
            list: List of 1s for yes, 0s for no
        """
        return [1 if self.compute_jaccard_similarity(u, v) > threshold else 0 for u, v in test_edges]

