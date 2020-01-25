from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np


""" Save all infos needed """
def save_model(clf, path='data/model.rf'):
    file = open(path, mode='w')

    print(len(clf.estimators_), file=file)
    print(len(clf.classes_), file=file)
    for e in clf.estimators_:
        tree = e.tree_
        print(tree.node_count, file=file)
        print(*tree.children_left, file=file)
        print(*tree.children_right, file=file)
        print(*tree.feature, file=file)
        print(*tree.threshold, file=file)
        for item in tree.value:
            print(*item.reshape(-1), file=file)

    file.close()


""" Save X in file"""
def save_data(X, path='data/X_test.data'):
    file = open(path, mode='w')

    for x in X:
        print(*x, file=file)
        
    file.close()

""" Save predict to file """
def save_predict(pred, path='data/python.pred'):
    file = open(path, mode='w')
    print(*pred, file=file, sep='\n')
    file.close()


if __name__ == "__main__":
    
    # Load dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create classifier
    clf = RandomForestClassifier(n_estimators=100)

    # rain classifier
    clf.fit(X_train, y_train)

    # Save the model
    save_model(clf)

    # Save the data to predict in C++
    save_data(X_test)

    # Save predict to compare with C++ predict
    save_predict(clf.predict(X_test))
    
