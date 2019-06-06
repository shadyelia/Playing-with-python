from sklearn.svm import SVC  # "Support vector classifier"
from sklearn import metrics


def SVMClasifier(attributes_train, attributes_test, labels_train, labels_test, CValue=1E10):
    """SVM Classifier print out the result"""

    # Training
    model = SVC(kernel='linear', decision_function_shape="ovr")
    model.fit(attributes_train, labels_train)

    # Predictions
    labels_pred = model.predict(attributes_test)

    print("Kernel = linear", "C value = default(1) ",
          'decision_function_shape = "ovr" ')
    print(metrics.accuracy_score(labels_test, labels_pred))

    model = SVC(decision_function_shape="ovr", gamma='scale', C=CValue)
    model.fit(attributes_train, labels_train)

    # Predictions
    labels_pred = model.predict(attributes_test)

    # Evaluating the Algorithm
    print("Kernel = RBF", "C value = " + str(CValue),
          ' decision_function_shape = "ovr" ')
    print(metrics.accuracy_score(labels_test, labels_pred))

    model = SVC(kernel='sigmoid', gamma='auto',
                decision_function_shape="ovr")
    model.fit(attributes_train, labels_train)

    # Predictions
    labels_pred = model.predict(attributes_test)

    # Evaluating the Algorithm
    print("Kernel = sigmoid", "C value = default(1)",
          ' decision_function_shape = "ovr" ')
    print(metrics.accuracy_score(labels_test, labels_pred))
