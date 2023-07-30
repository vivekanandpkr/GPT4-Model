from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Given sequences
sequences = [
    [4, 2, 1, 7, 3, 2, 7, 1],
    [1, 2, 7, 7, 3, 2, 7, 3],
    [5, 7, 3, 2, 7, 5, 7, 4],
    [7, 6, 7, 3, 2, 7, 4, 5],
    [2, 7, 3, 2, 7, 7, 5, 1],
    [1, 2, 7, 7, 3, 2, 7, 3],
    [5, 7, 3, 2, 7, 5, 7, 4],
    [7, 6, 7, 3, 2, 7, 4, 5],
    [2, 7, 3, 2, 7, 7, 5, 1]
]

# Prepare dataset
X, y = [], []
for sequence in sequences:
    for i in range(len(sequence) - 3):
        X.append(sequence[i:i+3])
        y.append(sequence[i+3])

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the model
model = MLPRegressor(random_state=42, max_iter=500)
model.fit(X_train, y_train)

# Test the model
print("Training score: ", model.score(X_train, y_train))
print("Testing score: ", model.score(X_test, y_test))

# Predict the next number in a new sequence
new_sequence = np.array([7, 1, 2]).reshape(1, -1)
print("Predicted next number: ", model.predict(new_sequence))
