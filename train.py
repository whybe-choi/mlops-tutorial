from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

iris = load_iris(as_frame=True)
X, y = iris['data'], iris['target']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=2024)

clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=2024)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_valid)
acc_score = accuracy_score(y_pred, y_valid)
print("Accuracy score is {:.4f}".format(acc_score))