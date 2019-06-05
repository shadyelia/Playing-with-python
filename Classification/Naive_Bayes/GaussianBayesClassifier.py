from sklearn.naive_bayes import GaussianNB


def GaussianClassifier(attributes_train, attributes_test, labels_train, labels_test):
    classifier = GaussianNB()
    classifier.fit(attributes_train, labels_train)
    # Predictions
    labels_pred = classifier.predict(attributes_test)
    # Evaluating the Algorithm
    acc = 0
    for i in range(len(labels_test)):
        if(labels_test[i] == labels_pred[i]):
            acc += 1
    print(acc / len(labels_test))
