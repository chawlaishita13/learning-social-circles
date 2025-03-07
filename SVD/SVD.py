import time
import numpy as np
from numpy.linalg import svd
from dataloader import SVDDataLoader
from itertools import combinations
from sklearn.metrics import accuracy_score, precision_score, recall_score

class SVDLinkPredictor:
    def __init__(self, adj_matrix, embedding_dim=50):
        self.adj_matrix = adj_matrix
        self.embedding_dim = embedding_dim
        self.U, self.S, self.Vt = self.compute_svd()
    
    def compute_svd(self):
        """Perform Singular Value Decomposition (SVD)."""
        U, S, Vt = svd(self.adj_matrix, full_matrices=False)
        return U[:, :self.embedding_dim], np.diag(S[:self.embedding_dim]), Vt[:self.embedding_dim, :]
    
    def predict_similarity(self, node1, node2):
        """Predict link existence using latent vector similarity."""
        vec1 = self.U[node1] @ self.S
        vec2 = self.U[node2] @ self.S
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
        
    def predict_links(self, test_edges, threshold=0.5):
        """Predict links using SVD-based similarity measure."""
        return [1 if self.predict_similarity(u, v) > threshold else 0 for u, v in test_edges]

# Load data
data_loader = SVDDataLoader(data_dir="Training")
adj_matrix = data_loader.adj_matrix
user_mapping = data_loader.user_mapping

# Generate test edges
all_possible_edges = list(combinations(range(len(user_mapping)), 2))
existing_edges = set(tuple(sorted([user_mapping[u], user_mapping[v]])) for u, v in data_loader.graph.edges())
test_edges = [edge for edge in all_possible_edges if edge not in existing_edges]
test_edges = test_edges[:min(len(existing_edges), len(test_edges))]  # Balance positive and negative samples

test_labels = [1 if edge in existing_edges else 0 for edge in test_edges]

# Train SVD Model
start_time = time.time()  # Start timing
model = SVDLinkPredictor(adj_matrix, embedding_dim=50)
predictions = model.predict_links(test_edges, threshold=0.5)
end_time = time.time()  # End timing

# Evaluate model
accuracy = accuracy_score(test_labels, predictions)
running_time = end_time - start_time

print(f"SVD Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Running Time: {running_time:.4f} seconds")
