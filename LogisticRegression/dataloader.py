import os
import networkx as nx
from utils import readcirclefile, readfeatures, readfeaturelist
import numpy as np
from tqdm import tqdm

class LogisticRegressionDataLoader:
    def __init__(self, data_dir: os.path):
        self.data_dir = data_dir
        self.data = None
        self.all_features = readfeaturelist(data_dir / 'featureList.txt')

    def load_data(self, requested_features: list):
        assert(type(requested_features) is list)
        for requested_feature in requested_features:
            if requested_feature not in self.all_features:
                raise KeyError(f"The requested feature {requested_feature} is not avaiable for the supplied dataset")

        profiles_dict = readfeatures(self.data_dir / 'features.txt')
        specific_profile = {}
        for profile in profiles_dict:
            id = profile['id'].pop()
            inisde_dict = {}
            for spec in requested_features:
                inisde_dict[spec] = profile.get(spec, set({-1})).pop()
            specific_profile[id] = inisde_dict

        graph = nx.Graph()
        trainingfiles = os.listdir(self.data_dir / 'Training')

        edges = []
        for item in trainingfiles:
            true_circles = readcirclefile(self.data_dir / 'Training' / item)
            for key in true_circles.keys():
                values = true_circles[key]
                for value in values:
                    edges.append((key, value))

        print(len(edges))
        graph.add_edges_from(edges)
        edges = list(graph.edges())
        non_edges = list(nx.non_edges(graph))

        user_combined_features = []
        data = []

        for user_id in tqdm(graph.nodes(), "Looping through graph nodes"):
            combined_features = list(specific_profile[user_id].values())
            user_combined_features.append(combined_features)

        print(len(user_combined_features))
        print(user_combined_features[0])
        for edge in tqdm(edges, "Looping through edges"):
            user1, user2 = edge
            label = 1  # They are friends
            try:
                features = np.concatenate([user_combined_features[user1], user_combined_features[user2]])
                data.append((features, label))
            except Exception as e:
                pass

        for non_edge in tqdm(non_edges, "Looping through non-edges"):
            user1, user2 = non_edge
            label = 0  # They are not friends :( sad
            try:
                features = np.concatenate([user_combined_features[user1], user_combined_features[user2]])
                data.append((features, label))
            except Exception as e:
                pass
        print(len(data))
        X = np.array([data_i[0] for data_i in data])
        y = np.array([data_i[1] for data_i in data])
        return X, y