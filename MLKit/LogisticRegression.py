import MLKit
import math
import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.thetas_dict = {}
        self.mean = {}
        self.std = {}
    
    def fit(self, X, Y, feature_names):
        np.seterr(all='raise')
        X = np.vstack([np.ones(X.shape[1]), X])
        thetas = np.zeros((np.unique(Y).shape[0], X.shape[0]))
        m = np.shape(Y)[0]
        linear_matrix = np.ones(m)

        for index, row_name in enumerate(np.unique(Y)):
            expected_results = np.where(Y == row_name, 1, 0)
            row_theta = thetas[index]
            prev_cost = -100
            cost = 0
            while abs(cost - prev_cost) > 0.00001:
                x = LogisticRegression.__g(row_theta.dot(X))
                x[x == 1] = 0.999
                prev_cost = cost
                cost = np.log(x) * expected_results + (1 - expected_results) * np.log(1 - x)
                cost = cost.dot(linear_matrix) / -m
                error = x - expected_results
                gradient = error * X
                new_thetas = linear_matrix.dot(gradient.T)
                new_thetas = new_thetas * self.learning_rate / m
                thetas[index] -= new_thetas

        for row_index, row_name in enumerate(np.unique(Y)):
            self.thetas_dict[row_name] = dict()
            self.thetas_dict[row_name]["t0"] = thetas[row_index][0]
            for feature_index, feature in enumerate(feature_names):
                self.thetas_dict[row_name][feature] = float(thetas[row_index][feature_index + 1])

    def save(self, file_name):
        MLKit.FileManager.save_model_data({**{"attributes": {"mean": self.mean, "std": self.std}}, **{"rows": self.thetas_dict}}, file_name)

    @staticmethod
    def predict(x):
        return LogisticRegression.__g(x)

    def __length(self, data, feature_column_names):
        length = {}

        for feature_name in feature_column_names:
            length[feature_name] = 0
            for row_name in data.keys():
                feature_data = data[row_name][feature_name]
                length[feature_name] += len(feature_data) - feature_data.count(None)

        return length

    @staticmethod
    def __g(z):
        try:
            return 1 / (1 + np.exp(-z))
        except FloatingPointError:
            MLKit.Display.error("Learning rate is too large")
