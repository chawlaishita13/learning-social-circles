import numpy as np
from numpy.linalg import svd

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
