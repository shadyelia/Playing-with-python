from sklearn.svm import SVC  # "Support vector classifier"


def SVMClasifier(attributes_train, attributes_test, labels_train, labels_test, CValue=1E10):
    """SVM Classifier print out the result"""

    # Training
    model = SVC(kernel='linear', decision_function_shape="ovr")
    model.fit(attributes_train, labels_train)

    # Predictions
    labels_pred = model.predict(attributes_test)

    # Evaluating the Algorithm
    acc = 0
    for i in range(len(labels_test)):
        if(labels_test[i] == labels_pred[i]):
            acc += 1

    print("Kernel = linear", "C value = default(1) ",
          'decision_function_shape = "ovr" ')
    print(acc / len(labels_test))

    model = SVC(decision_function_shape="ovr", gamma='scale', C=CValue)
    model.fit(attributes_train, labels_train)

    # Predictions
    labels_pred = model.predict(attributes_test)

    # Evaluating the Algorithm
    acc = 0
    for i in range(len(labels_test)):
        if(labels_test[i] == labels_pred[i]):
            acc += 1

    print("Kernel = RBF", "C value = " + str(CValue),
          ' decision_function_shape = "ovr" ')
    print(acc / len(labels_test))

    model = SVC(kernel='sigmoid', gamma='auto',
                decision_function_shape="ovr")
    model.fit(attributes_train, labels_train)

    # Predictions
    labels_pred = model.predict(attributes_test)

    # Evaluating the Algorithm
    acc = 0
    for i in range(len(labels_test)):
        if(labels_test[i] == labels_pred[i]):
            acc += 1

    print("Kernel = sigmoid", "C value = default(1)",
          ' decision_function_shape = "ovr" ')
    print(acc / len(labels_test))
