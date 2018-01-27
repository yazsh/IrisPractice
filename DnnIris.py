import tensorflow as tf
import pandas as pd
import numpy as np


def input_function(features,labels,shuffle=False):

    # labels = tf.one_hot(labels,depth=3)
    labels = tf.convert_to_tensor(labels)
    return {"sepallength": tf.convert_to_tensor(features['sepallength']),
            "sepalwidth": tf.convert_to_tensor(features['sepalwidth']),
            "petallength": tf.convert_to_tensor(features['petallength']),
            "petalwidth": tf.convert_to_tensor(features['petalwidth'])}, labels

def serving_input_receiver_fn():
    serialized_tf_example = tf.placeholder(dtype=tf.string,
                                           shape=[1],
                                           name='input_example_tensor')
    feature_spec = {'petallength': tf.FixedLenFeature(shape=[1], dtype=tf.float32),
                    'sepalwidth': tf.FixedLenFeature(shape=[1], dtype=tf.float32),
                    'petalwidth': tf.FixedLenFeature(shape=[1], dtype=tf.float32),
                    'sepallength': tf.FixedLenFeature(shape=[1], dtype=tf.float32)}

def serving_input_fn():
    inputs = {'petallength': tf.placeholder(tf.float32, [1]),
              'sepalwidth': tf.placeholder(tf.float32, [1]),
              'petalwidth': tf.placeholder(tf.float32, [1]),
              'sepallength': tf.placeholder(tf.float32, [1])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

train = pd.read_csv("/Users/yazen/Desktop/datasets/iris/iris_training.csv")
test = pd.read_csv("/Users/yazen/Desktop/datasets/iris/iris_test.csv")

train.columns = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species']
test.columns = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species']

train_features = train.drop('species', axis=1)
test_features = test.drop('species', axis=1)

train_labels = train['species']
test_labels = test['species']

feature_columns = train_features.columns

my_feature_columns = []
for key in train_features.columns:
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))


classifier = tf.estimator.DNNRegressor(
      feature_columns=my_feature_columns, hidden_units=[10, 14, 12, 10])

classifier.train(input_fn=lambda:input_function(train_features, train_labels,False), steps=400 )

print("evaluating")

classifier.export_savedmodel("/Users/yazen/Desktop/mlprojects/Iris",serving_input_fn)


print("evaluating")
# print(classifier.evaluate(input_fn=lambda:input_function(test_features, test_labels, False)))