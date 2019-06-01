from sklearn.svm import SVC  # "Support vector classifier"


def SVMClasifier(attributes_train, attributes_test, labels_train, labels_test, CValue=1E10):
    """SVM Classifier print out the result"""

    # Training
    model = SVC(kernel='linear', decision_function_shape="ovr", C=CValue)
    model.fit(attributes_train, labels_train)

    # Predictions
    labels_pred = model.predict(attributes_test)

    # Evaluating the Algorithm
    acc = 0
    for i in range(labels_test.size):
        if(labels_test[i] == labels_pred[i]):
            acc += 1

    print("Kernel = linear", "C value = 1E10 ",
          'decision_function_shape = "ovr" ')
    print(acc / labels_test.size)

    model = SVC(decision_function_shape="ovr", C=CValue)
    model.fit(attributes_train, labels_train)

    # Predictions
    labels_pred = model.predict(attributes_test)

    # Evaluating the Algorithm
    acc = 0
    for i in range(labels_test.size):
        if(labels_test[i] == labels_pred[i]):
            acc += 1

    print("Kernel = RBF", "C value = 1E10 ",
          'decision_function_shape = "ovr" ')
    print(acc / labels_test.size)

    model = SVC(kernel='sigmoid', decision_function_shape="ovr", C=CValue)
    model.fit(attributes_train, labels_train)

    # Predictions
    labels_pred = model.predict(attributes_test)

    # Evaluating the Algorithm
    acc = 0
    for i in range(labels_test.size):
        if(labels_test[i] == labels_pred[i]):
            acc += 1

    print("Kernel = sigmoid", "C value = 1E10 ",
          'decision_function_shape = "ovr" ')
    print(acc / labels_test.size)
