from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def KnnClasifier(dataset, N=5):
    """KNN Classifier print out the result"""
    # Preprocessing
    # 1. split our dataset into its attributes and labels
    attributes = dataset.iloc[:, :-1].values
    labels = dataset.iloc[:, 4].values

    header_attributes = attributes[0]
    header_labels = labels[0]

    # 2. divide our dataset into training and test splits
    attributes_train, attributes_test, labels_train, labels_test = train_test_split(
        attributes[1:], labels[1:], test_size=0.20)

    # 3.Feature Scaling (normalization)
    scaler = StandardScaler()
    scaler.fit(attributes_train)

    attributes_train = scaler.transform(attributes_train)
    attributes_test = scaler.transform(attributes_test)

    # Training
    classifier = KNeighborsClassifier(n_neighbors=N)
    classifier.fit(attributes_train, labels_train)

    # Predictions
    labels_pred = classifier.predict(attributes_test)

    # Evaluating the Algorithm
    print(confusion_matrix(labels_test, labels_pred))
    print(classification_report(labels_test, labels_pred))

    error = []
    # Calculating error for K values between 1 and 40
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(attributes_train, labels_train)
        pred_i = knn.predict(attributes_test)
        error.append(np.mean(pred_i != labels_test))

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()
