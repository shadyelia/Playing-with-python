from KNN.KNN import KnnClasifier
import pandas as pd
from SVM.SVM import SVMClasifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Naive_Bayes.GaussianBayesClassifier import GaussianClassifier


def main():
    """Classifcation to IRIS data set with different ways"""
    path = 'IRIS.csv'

    # Assign colum names to the dataset
    names = ['sepal-length', 'sepal-width',
             'petal-length', 'petal-width', 'Class']

    # Read dataset to pandas dataframe
    dataset = pd.read_csv(path, names=names)

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

    print('1-KNN : ')
    KnnClasifier(attributes_train, attributes_test,
                 labels_train, labels_test, N=6)

    print('2-SVM : ')
    SVMClasifier(attributes_train, attributes_test, labels_train, labels_test)

    print('3-Bayse : ')
    GaussianClassifier(attributes_train, attributes_test,
                       labels_train, labels_test)


if __name__ == '__main__':
    main()
