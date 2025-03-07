from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from dataloader import LogisticRegressionDataLoader
from pathlib import Path

labels = ['locale', 'education;school;id', 'education;year;id', 'work;employer;id']

root_dir = Path.cwd().resolve()
data_dir = root_dir / 'data'
data_loader = LogisticRegressionDataLoader(data_dir=data_dir)

all_X, all_y = data_loader.load_data(labels)

X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.3, random_state=74)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy on the test set: {accuracy * 100:.2f}%")