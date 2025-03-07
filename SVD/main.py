from dataloader import SVDDataLoader
from itertools import combinations
from sklearn.metrics import accuracy_score
from pathlib import Path
import time
from SVD import SVDLinkPredictor

if __name__ == '__main__':
    root_dir = Path.cwd().resolve()
    svd_dir = root_dir / 'SVD'
    # Load data
    data_loader = SVDDataLoader(data_dir=svd_dir / "Training")
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
