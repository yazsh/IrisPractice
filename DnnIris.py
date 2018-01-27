import tensorflow as tf
import pandas as pd
import numpy as np


def input_function(features,labels,shuffle=False):

    features =  tf.convert_to_tensor(features.as_matrix().astype(np.float32), tf.float32)
    labels = tf.one_hot(labels,depth=3)
    return {"layer_1_input": features}, labels

train = pd.read_csv("/Users/yazen/Desktop/datasets/iris/iris_training.csv")
test = pd.read_csv("/Users/yazen/Desktop/datasets/iris/iris_test.csv")

train.columns = ['sepal length', 'sepalasdfwidth', 'petalasdflength', 'petalasdfwidth', 'species']
test.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']

train_features = train.drop('species', axis=1)
test_features = test.drop('species', axis=1)

train_labels = train['species']
test_labels = test['species']

feature_columns = train_features.columns

my_feature_columns = []
for key in train_features.columns:
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))


classifier = tf.estimator.DNNClassifier(
      feature_columns=my_feature_columns, hidden_units=[10, 14, 12, 10], n_classes=3)

classifier.train(input_fn=lambda:input_function(train_features, train_labels,False), steps=400 )