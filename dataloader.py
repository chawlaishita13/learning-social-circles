import os
import numpy as np
import tempfile
import random


class DataLoader:
    def __init__(self, training_dir, target_per_class=100000):
        self._training_dir = training_dir

        # Load raw circles and users data.
        self._circles, self._users = load_data(training_dir)

        # Encode user mappings.
        self._encoded_users_mapping, self._user_encoding, self._member_encoding = encode_user_mappings(self._users)

        # Map users to encoded connected members.
        self._encoded_users = map_users_to_connected_members(self._users, self._user_encoding, self._member_encoding)

        # Build encoded user vectors.
        self._encoded_user_vectors = build_encoded_user_vectors(self._encoded_users, self._member_encoding)

        # Create training data.
        self._X, self._y = create_training_data(self._encoded_user_vectors, target_per_class=target_per_class)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_spilt(self._X, self._y)


def load_data(training_dir):
    # Dictionary to store circle membership.
    # Keys are circle IDs and values are sets of user IDs.
    circles = {}

    # Keys are user IDs and values are sets of members in the circles the user is in.
    users = {}

    # Process all *.circles files in the Training directory.
    for filename in os.listdir(training_dir):
        if filename.endswith(".circles"):
            user_id = filename[:-8]  # remove ".circles"
            file_path = os.path.join(training_dir, filename)
            
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Expect each line to be formatted as "circleID: memberID1 memberID2 ..."
                    if ":" in line:
                        circle_id, members_str = line.split(":", 1)
                        circle_id = circle_id.strip()
                        members = members_str.split()  # split the remaining tokens
                        
                        if circle_id not in circles:
                            circles[circle_id] = set()
                            
                        if user_id not in users:
                            users[user_id] = set()
                            
                        circles[circle_id].update(members)
                        users[user_id].update(members)

    # Optional: Convert sets to lists for easier viewing/manipulation.
    circles = {cid: list(members) for cid, members in circles.items()}
    users = {uid: list(members) for uid, members in users.items()}
    
    return circles, users

def encode_user_mappings(users):
    # Create encoding mapping for users by sorting user IDs as integers.
    user_ids = sorted(users.keys(), key=lambda x: int(x))
    user_encoding = {uid: i for i, uid in enumerate(user_ids)}

    # Create a new dictionary to store encoded user data
    encoded_users = {}
    
    # Create encoding mapping for members (0 to 11544) based on the sorted list all_members.
    combined_members = set()
    for member_list in users.values():
        combined_members.update(member_list)
    all_members = sorted(combined_members)
    member_encoding = {m: i for i, m in enumerate(all_members)}

    for uid, member_list in users.items():
        # Map the user id using the user_encoding
        encoded_uid = user_encoding.get(uid, uid)
        encoded_users[encoded_uid] = member_list

    return encoded_users, user_encoding, member_encoding

def map_users_to_connected_members(users, user_encoding, member_encoding):
    encoded_users = {}

    for uid, member_list in users.items():
        # Map the user id using the user_encoding
        encoded_uid = user_encoding.get(uid, uid)
        
        # For each member in the user's list, map using member_encoding
        encoded_member_list = [member_encoding[m] for m in member_list if m in member_encoding]
        
        encoded_users[encoded_uid] = encoded_member_list
        
    return encoded_users

def build_encoded_user_vectors(encoded_users, member_encoding):
    # Build a mapping from each encoded user to a vector of length equal to the number of encoded members
    encoded_user_vectors = {}

    for uid, encoded_member_list in encoded_users.items():
        member_set = set(encoded_member_list)
        vector = np.array([1 if i in member_set else -1 for i in range(len(member_encoding))])
        encoded_user_vectors[uid] = vector

    return encoded_user_vectors

def create_training_data(encoded_user_vectors, target_per_class=100000):

    # Define cache folder and file path.
    cache_folder = "cache"
    cache_file = os.path.join(cache_folder, f"training_data_{target_per_class}.npz")

    # If cached data exists, load and return.
    if os.path.exists(cache_file):
        print("Loaded cached training data")
        data = np.load(cache_file)
        X = data['X']
        y = data['y']
        return X, y

    print("Cached training data not found. Creating new training data.")

    positive_points = []
    negative_points = []

    # Get a list of (uid, vector) pairs
    users_list = list(encoded_user_vectors.items())

    # Sample points until we have enough in each class.
    while len(positive_points) < target_per_class or len(negative_points) < target_per_class:
        for uid, vector in users_list:
            if len(positive_points) >= target_per_class and len(negative_points) >= target_per_class:
                break

            vec = vector.copy()
            pos_indices = (vec == 1).nonzero()[0]
            neg_indices = (vec == -1).nonzero()[0]

            if len(positive_points) < target_per_class and len(pos_indices) > 0:
                idx = random.choice(pos_indices)
                vec_masked = vec.copy()
                # Mask the positive index
                vec_masked[idx] = 0  
                positive_points.append((uid, vec_masked, 1))

            if len(negative_points) < target_per_class and len(neg_indices) > 0:
                idx = random.choice(neg_indices)
                vec_masked = vec.copy()
                # Mask the negative index
                vec_masked[idx] = 0  
                negative_points.append((uid, vec_masked, 0))

    # Combine the points and shuffle
    training_points = positive_points[:target_per_class] + negative_points[:target_per_class]
    random.shuffle(training_points)

    # Split features (X) and labels (y)
    X = np.array([vec for uid, vec, label in training_points])
    y = np.array([label for uid, vec, label in training_points])

    # Ensure the cache directory exists and save the training data.
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    np.savez_compressed(cache_file, X=X, y=y)

    return X, y

def train_test_spilt(X,y):
    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':

    # Create a temporary directory and dummy *.circles files for testing.
    with tempfile.TemporaryDirectory() as temp_dir:
        dummy_files = {
            "1.circles": "100: 200 300\n101: 300\n",
            "2.circles": "100: 300 400\n",
            "3.circles": "102: 200 400\n"
        }
        for fname, content in dummy_files.items():
            with open(os.path.join(temp_dir, fname), "w") as f:
                f.write(content)

        # Load circles and users from the dummy data.
        print("\nTesing function: load_data")
        circles, users = load_data(temp_dir)
        print("Circles sample:\n", list(circles.items())[:2])
        print("Users sample:\n", list(users.items())[:2])
        print("-"*20)

        # Build a user encoding mapping (sorted by int conversion).
        print("\nTesing function: encode_user_mappings")
        encoded_users_mapping, user_encoding, member_encoding = encode_user_mappings(users)
        print("Encoded users mapping sample:\n", list(encoded_users_mapping.items())[:2])
        print("-"*20)

        # Map users to encoded connected members.
        print("\nTesing function: map_users_to_connected_members")
        encoded_users = map_users_to_connected_members(users, user_encoding, member_encoding)
        print("Encoded users sample:\n", list(encoded_users.items())[:2])
        print("-"*20)

        # Build encoded user vectors.
        print("\nTesing function: build_encoded_user_vectors")
        encoded_user_vectors = build_encoded_user_vectors(encoded_users, member_encoding)
        for uid, vector in encoded_user_vectors.items():
            print(f"User vector for {uid} (first 10 elements):\n", vector[:10])
        print("-"*20)

        # Create training data using a small target_per_class for testing.
        print("\nTesing function: create_training_data")
        X, y = create_training_data(encoded_user_vectors, target_per_class=1)
        print("Training data features sample:\n", X[:2])
        print("Training data labels sample:\n", y[:2])
        print("-"*20)