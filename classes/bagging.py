import warnings
warnings.simplefilter("ignore")
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    dt_heart = pd.read_csv("./datasets/heart.csv")
    print(dt_heart['target'].describe())
    
    X = dt_heart.drop(['target'], axis=1)
    y = dt_heart['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
    
    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_pred = knn_class.predict(X_test)
    print("="*64)
    print('KNeighbors sin bagging', accuracy_score(knn_pred, y_test))
    
    # bag_class = BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train)
    # bag_pred = bag_class.predict(X_test)
    # print("="*64)
    # print(accuracy_score(bag_pred, y_test))
    
    classifiers = {
        'KNeighbors': KNeighborsClassifier(),
        'LinearSCV': LinearSVC(),
        'SVC': SVC(),
        'SGDC': SGDClassifier(),
        'DecisionTree': DecisionTreeClassifier()
    }
    
    for name, estimador in classifiers.items():
        bag_class = BaggingClassifier(estimator=estimador, n_estimators=5).fit(X_train, y_train)
        bag_pred = bag_class.predict(X_test)
        
        print("="*64) 
        print('Accuracy Bagging with {}:'.format(name), accuracy_score(bag_pred, y_test))
        