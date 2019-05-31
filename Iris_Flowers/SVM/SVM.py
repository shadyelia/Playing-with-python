from sklearn.svm import SVC  # "Support vector classifier"
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def SVMClasifier(dataset, CValue=1E10):
    """SVM Classifier print out the result"""
    # Preprocessing
    # 1. split our dataset into its attributes and labels
    attributes = dataset.iloc[:, :-1].values
    labels = dataset.iloc[:, 4].values

    # header_attributes = attributes[0]
    # header_labels = labels[0]

    # 2. divide our dataset into training and test splits
    attributes_train, attributes_test, labels_train, labels_test = train_test_split(
        attributes[1:], labels[1:], test_size=0.20)

    # 3.Feature Scaling (normalization)
    scaler = StandardScaler()
    scaler.fit(attributes_train)

    attributes_train = scaler.transform(attributes_train)
    attributes_test = scaler.transform(attributes_test)

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
