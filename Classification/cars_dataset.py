from KNN.KNN import KnnClasifier
import pandas as pd
from SVM.SVM import SVMClasifier
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def main():
    """Classifcation to IRIS data set with different ways"""
    path = 'Classification\\cars_dataset.csv'

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
    numericAttributes = []
    numericLabels = []
    for label in labels[1:]:
        if label == 'unacc':
            numericLabels.append(0)
        elif label == 'acc':
            numericLabels.append(1)
        elif label == 'good':
            numericLabels.append(2)
        elif label == 'vgood':
            numericLabels.append(3)

    for attribute in attributes[1:]:
        numericAttribute = []
        if(attribute[0] == 'low'):
            numericAttribute.append(0)
        elif (attribute[0] == 'med'):
            numericAttribute.append(1)
        elif(attribute[0] == 'high'):
            numericAttribute.append(2)
        elif (attribute[0] == 'vhigh'):
            numericAttribute.append(3)

        if(attribute[1] == 'low'):
            numericAttribute.append(0)
        elif (attribute[1] == 'med'):
            numericAttribute.append(1)
        elif(attribute[1] == 'high'):
            numericAttribute.append(2)
        elif (attribute[1] == 'vhigh'):
            numericAttribute.append(3)

        if(attribute[2] == 'two'):
            numericAttribute.append(0)
        elif (attribute[2] == 'three'):
            numericAttribute.append(1)
        elif(attribute[2] == 'four'):
            numericAttribute.append(2)
        elif (attribute[2] == '5more'):
            numericAttribute.append(3)

        if(attribute[3] == 'two'):
            numericAttribute.append(0)
        elif (attribute[3] == 'four'):
            numericAttribute.append(1)
        elif(attribute[3] == 'more'):
            numericAttribute.append(2)

        if(attribute[4] == 'small'):
            numericAttribute.append(0)
        elif (attribute[4] == 'med'):
            numericAttribute.append(1)
        elif(attribute[4] == 'big'):
            numericAttribute.append(2)

        if(attribute[5] == 'low'):
            numericAttribute.append(0)
        elif (attribute[5] == 'med'):
            numericAttribute.append(1)
        elif(attribute[5] == 'high'):
            numericAttribute.append(2)

        numericAttributes.append(numericAttribute)

    # 3. divide our dataset into training and test splits
    attributes_train, attributes_test, labels_train, labels_test = train_test_split(
        numericAttributes, numericLabels, test_size=0.20)

    print('1-KNN : ')
    KnnClasifier(attributes_train, attributes_test,
                 labels_train, labels_test, N=4)

    print('2-SVM : ')
    SVMClasifier(attributes_train, attributes_test,
                 labels_train, labels_test)


if __name__ == '__main__':
    main()
