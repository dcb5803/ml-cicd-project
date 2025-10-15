# tests/test_model.py
import pytest
from model import SimpleModel
import numpy as np

def test_model_training_and_prediction():
    """
    Tests if the model can be trained and if the prediction output shape is correct.
    """
    # Arrange: Create model and dummy data
    model = SimpleModel()
    X_train = np.array([[1], [2], [3]])
    y_train = np.array([2, 4, 6])
    X_test = np.array([[4], [5]])

    # Act: Train the model and make a prediction
    model.train(X_train, y_train)
    predictions = model.predict(X_test)

    # Assert: Check if the output has the expected shape (2 predictions)
    assert predictions.shape == (2,)
