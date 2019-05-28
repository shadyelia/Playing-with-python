from KNN.KNN import KnnClasifier
import pandas as pd


def main():
    """Classifcation to IRIS data set with different ways"""
    path = 'Iris_Flowers\\IRIS.csv'

    # Assign colum names to the dataset
    names = ['sepal-length', 'sepal-width',
             'petal-length', 'petal-width', 'Class']

    # Read dataset to pandas dataframe
    dataset = pd.read_csv(path, names=names)

    print('1-KNN : ')
    KnnClasifier(dataset, N=6)


if __name__ == '__main__':
    main()
