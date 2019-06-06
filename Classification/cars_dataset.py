from KNN.KNN import KnnClasifier
import pandas as pd
from SVM.SVM import SVMClasifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Naive_Bayes.GaussianBayesClassifier import GaussianClassifier
from sklearn import preprocessing


def main():
    """Classifcation to cars dataset set with different ways"""
    path = 'cars_dataset.csv'

    # Assign colum names to the dataset
    names = ['buying', 'maint',
             'doors', 'persons', 'lug_boot', 'safety', 'car']

    # Read dataset to pandas dataframe
    dataset = pd.read_csv(path, names=names)

    # Preprocessing
    # 1. split our dataset into its attributes and labels
    attributes = dataset.iloc[:, :-1].values
    labels = dataset.iloc[:, 6].values

    # 2. convert data to numeric
    le = preprocessing.LabelEncoder()
    buying, maint, doors, persons, lug_boot, safety = attributes[:,
                                                                 0], attributes[:, 1], attributes[:, 2], attributes[:, 3], attributes[:, 4], attributes[:, 5]
    numericAttributes = zip(le.fit_transform(buying[1:]), le.fit_transform(maint[1:]), le.fit_transform(
        doors[1:]), le.fit_transform(persons[1:]), le.fit_transform(lug_boot[1:]), le.fit_transform(safety[1:]))
    numericLabels = le.fit_transform(labels[1:])

    # 3. divide our dataset into training and test splits
    attributes_train, attributes_test, labels_train, labels_test = train_test_split(
        list(numericAttributes), list(numericLabels), test_size=0.20)

    # 4.Feature Scaling (normalization)
    scaler = StandardScaler()
    scaler.fit(attributes_train)

    attributes_train = scaler.transform(attributes_train)
    attributes_test = scaler.transform(attributes_test)

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
