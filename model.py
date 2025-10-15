# model.py
from sklearn.linear_model import LinearRegression
import numpy as np

class SimpleModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        """Trains the model."""
        self.model.fit(X, y)
        print("Model trained successfully!")

    def predict(self, X):
        """Makes predictions."""
        return self.model.predict(X)
