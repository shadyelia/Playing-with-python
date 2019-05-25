from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

path = 'Iris_Flowers\\IRIS.csv'

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(path, names=names)


# Preprocessing
# 1. split our dataset into its attributes and labels
attributes = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, 4].values

header_attributes = attributes[0]
header_labels = labels[0]

# 2. divide our dataset into training and test splits
attributes_train, attributes_test, labels_train, labels_test = train_test_split(
    attributes[1:], labels[1:], test_size=0.20)

# 3.Feature Scaling
scaler = StandardScaler()
scaler.fit(attributes_train)

attributes_train = scaler.transform(attributes_train)
attributes_test = scaler.transform(attributes_test)

# Training
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(attributes_train, labels_train)

# Predictions
labels_pred = classifier.predict(attributes_test)

# Evaluating the Algorithm
print(confusion_matrix(labels_test, labels_pred))
print(classification_report(labels_test, labels_pred))
