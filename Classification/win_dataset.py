from KNN.KNN import KnnClasifier
import pandas as pd
from SVM.SVM import SVMClasifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Naive_Bayes.GaussianBayesClassifier import GaussianClassifier
from sklearn import preprocessing
from sklearn import datasets


def main():
    """Classifcation to wine dataset set with different ways"""
    wine = datasets.load_wine()

    # Preprocessing
    # 1. split our dataset into its attributes and labels
    attributes = wine["data"]
    labels = wine["target"]

    # 2. divide our dataset into training and test splits
    attributes_train, attributes_test, labels_train, labels_test = train_test_split(
        attributes, labels, test_size=0.20)

    print('1-KNN : ')
    KnnClasifier(attributes_train, attributes_test,
                 labels_train, labels_test, N=4)

    print('2-SVM : ')
    SVMClasifier(attributes_train, attributes_test,
                 labels_train, labels_test)

    print('3-Bayse : ')
    GaussianClassifier(attributes_train, attributes_test,
                       labels_train, labels_test)


if __name__ == '__main__':
    main()
