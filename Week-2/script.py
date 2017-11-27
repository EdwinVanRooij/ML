import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib
import matplotlib.pyplot as plt

print('numpy version:', np.__version__)
print('pandas version:', pd.__version__)
print('scikit-learn version:', sk.__version__)
print('matplotlib version:', matplotlib.__version__)

df = pd.read_csv("ML02_Iris.data")

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Label the data by species
df['Label'] = df['Species'].map({'Iris-virginica': 0, 'Iris-versicolor': 3, 'Iris-setosa': 5})

# print("Head:")
# print(df.head(5))
#
# print("\n\nMiddle:")
# print(df.iloc[range(75, 80)])
#
# print("\n\nTail:")
# print(df.tail(5))

from sklearn.cross_validation import train_test_split

# We'll use only two features to train this algorithm
features = df[['Sepal Length', 'Sepal Width']]

# Get the label which corresponds to a species
labels = df['Label']

# Split the dataset into training and testing data
features_training, features_tests, labels_training, labels_tests = train_test_split(features, labels, test_size=.25)
print("Features training: {}, labels training: {}\nFeatures tests: {}, labels tests: {}".format(features_training.shape,
                                                                                                labels_training.shape,
                                                                                                features_tests.shape,
                                                                                                labels_tests.shape))
from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler().fit(features_training)
features_training = scaler.transform(features_training)
features_tests = scaler.transform(features_tests)

colors = ('red', 'greenyellow', 'blue')
for i in range(len(colors)):
    x1s = features_training[:, 0][labels_training.as_matrix() == i]
    x2s = features_training[:, 1][labels_training.as_matrix() == i]
    plt.scatter(x1s, x2s, c=colors[i])

plt.legend(['setosa', 'versicolor', 'virginica'])   # maintain order from dataset preparation!
plt.xlabel(features.columns[0])
plt.ylabel(features.columns[1])

# Create the linear model classifier
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(features_training, labels_training)

# Print the 'learned' coefficients
print("Coef: {}".format(clf.coef_))
print("Intercept: {}".format(clf.intercept_))
