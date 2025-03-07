from JaccardSimilarity import JaccardSimilarityModel
from dataloader import JaccardDataLoader
from pathlib import Path
import time
from itertools import combinations
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    root_dir = Path.cwd().resolve()
    jaccard_path = root_dir / 'JaccardSimilarity'

    # Load data
    data_loader = JaccardDataLoader(data_dir=jaccard_path / "Training")
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