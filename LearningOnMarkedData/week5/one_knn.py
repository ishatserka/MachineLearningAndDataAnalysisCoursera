import numpy as np
from sklearn import datasets, metrics
from pybrain3.utilities import percentError
from sklearn.neighbors import KNeighborsClassifier

class OneNNClassifier:
    _X = None
    _y = None

    def __init__(self):
        self._X = None
        self._y = None

    def fit(self, X_train, y_train):
        self._X = np.array(X_train)
        self._y = np.array(y_train)

    def predict(self, X_test):
        y_predict = list()
        for i in range(0, np.array(X_test).shape[0]):
            predict = list()
            for j in range(0, self._X.shape[0]):
                predict.append([self.__evclid_metric__(self._X[j, :], X_test[i, :]), self._y[j]])
            predict = sorted(predict)
            y_predict.append(predict[0][1])
        return y_predict

    def __evclid_metric__(self, train, test):
        res = train - test
        return sum(map(lambda x: x**2, res))


if __name__ == "__main__":
    digits = datasets.load_digits()
    X_train, X_test = digits.data[: 1345, :], digits.data[1346:, :]
    y_train, y_test = digits.target[: 1345], digits.target[1346:]
    estimator = OneNNClassifier()
    estimator.fit(X_train, y_train)
    y_predict = estimator.predict(X_test)

    print(1 - metrics.accuracy_score(y_test, y_predict))

    one_clf = KNeighborsClassifier(n_neighbors=1, weights='distance')
    one_clf.fit(X_train, y_train)

    one_predicted = one_clf.predict(X_test)
    print(1 - metrics.accuracy_score(y_test, one_predicted))
