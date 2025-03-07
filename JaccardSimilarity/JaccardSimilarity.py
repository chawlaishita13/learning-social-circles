import time
import networkx as nx
from dataloader import JaccardDataLoader
from itertools import combinations
from sklearn.metrics import accuracy_score

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
        return [1 if self.compute_jaccard_similarity(u, v) > threshold else 0 for u, v in test_edges]

# Load data
data_loader = JaccardDataLoader(data_dir="Training")
graph = data_loader.graph

# Generate test edges (Non-existing links for evaluation)
all_possible_edges = list(combinations(graph.nodes, 2))
existing_edges = set(graph.edges())
test_edges = [edge for edge in all_possible_edges if edge not in existing_edges]
test_edges = test_edges[:min(len(existing_edges), len(test_edges))]  # Balance positive and negative samples

test_labels = [1 if edge in existing_edges else 0 for edge in test_edges]

# Run Jaccard Similarity Model
model = JaccardSimilarityModel(graph)

start_time = time.time()  # Start timing
predictions = model.predict_links(test_edges, threshold=0.01)
end_time = time.time()  # End timing

# Evaluate model
accuracy = accuracy_score(test_labels, predictions)
running_time = end_time - start_time

print(f"Jaccard Similarity Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Running Time: {running_time:.4f} seconds")