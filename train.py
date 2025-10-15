# train.py
from model import SimpleModel
import numpy as np

# Create dummy data
X_train = np.array([[1], [2], [3], [4]])
y_train = np.array([2, 4, 6, 8])

# Initialize and train the model
model = SimpleModel()
model.train(X_train, y_train)

# Make a prediction
prediction = model.predict(np.array([[5]]))
print(f"Prediction for input [[5]]: {prediction[0]}")
