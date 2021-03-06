from KNN.KNN import KnnClasifier
import pandas as pd
from SVM.SVM import SVMClasifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Naive_Bayes.GaussianBayesClassifier import GaussianClassifier
from Naive_Bayes.GaussianBayesClassifier_implement import GaussianClassifier_Implement


def main():
    """Classifcation to IRIS data set with different ways"""
    path = 'pima_indians_diabetes.data.csv'

    # Read dataset to pandas dataframe
    dataset = pd.read_csv(path)

    # Preprocessing
    # 1. split our dataset into its attributes and labels
    attributes = dataset.iloc[:, :-1].values
    labels = dataset.iloc[:, 8].values

    # 2. divide our dataset into training and test splits
    attributes_train, attributes_test, labels_train, labels_test = train_test_split(
        attributes, labels, test_size=0.20)

    # 3.Feature Scaling (normalization)
    scaler = StandardScaler()
    scaler.fit(attributes_train)

    attributes_train = scaler.transform(attributes_train)
    attributes_test = scaler.transform(attributes_test)

    print('1-KNN : ')
    KnnClasifier(attributes_train, attributes_test,
                 labels_train, labels_test, N=2)

    print('2-SVM : ')
    SVMClasifier(attributes_train, attributes_test,
                 labels_train, labels_test, CValue=1)

    print('3-Gaussian Bayes : ')
    GaussianClassifier(attributes_train, attributes_test,
                       labels_train, labels_test)

    print('4-Gaussian Bayes (implemented) : ')
    GaussianClassifier_Implement(dataset.values)


if __name__ == '__main__':
    main()
