import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the Iris dataset
dataset = load_iris()

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(dataset["data"], dataset["target"], random_state=0)

# Initialize a list to store error rates for different values of k
error_rates = []

# Loop through different values of k (number of neighbors)
k_range = range(1, 41)  # We will test k from 1 to 40
for k in k_range:
    # Create and train the KNN classifier with the current value of k
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict the species of the test samples
    y_pred = knn.predict(X_test)

    # Print target and predicted species for each test sample
    print(f"\nFor k = {k} neighbors:")
    for i in range(len(X_test)):
        print(f"TARGET = {dataset['target_names'][y_test[i]]} | PREDICTED = {dataset['target_names'][y_pred[i]]}")

    # Calculate the error rate for the current k value
    accuracy = knn.score(X_test, y_test)
    error_rate = 1 - accuracy
    error_rates.append(error_rate)

# Plot the error rates for different values of k
plt.figure(figsize=(10, 6))
plt.plot(k_range, error_rates, marker='o', color='b', linestyle='-', markersize=6)
plt.title('Error Rate vs Number of Neighbors (k) in KNN Classifier')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Error Rate')
plt.show()

print('Final accuracy =', accuracy)
print('Final error_rate =', error_rate)
