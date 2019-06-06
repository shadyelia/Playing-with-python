from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


def GaussianClassifier(attributes_train, attributes_test, labels_train, labels_test):
    classifier = GaussianNB()
    classifier.fit(attributes_train, labels_train)
    # Predictions
    labels_pred = classifier.predict(attributes_test)
    # Evaluating the Algorithm
    print("Accuracy:", metrics.accuracy_score(labels_test, labels_pred))
