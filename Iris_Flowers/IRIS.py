import KNN.KNN
import pandas as pd

path = 'Iris_Flowers\\IRIS.csv'

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(path, names=names)

KNN.KNN.KnnClasifier(dataset)
