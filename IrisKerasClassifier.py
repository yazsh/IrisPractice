import tensorflow as tf
import pandas as pd
import numpy as np

train = pd.read_csv("/Users/yazen/Desktop/datasets/iris/iris_training.csv")
test = pd.read_csv("/Users/yazen/Desktop/datasets/iris/iris_test.csv")

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(10, input_shape=(4,), activation='relu',name="layer_1"))
model.add(tf.keras.layers.Dense(14, activation='relu',))
model.add(tf.keras.layers.Dense(12, activation='relu', ))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

tf_model = tf.keras.estimator.model_to_estimator(model)
train.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']
test.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']

train_features = train.drop('species', axis=1)
test_features = test.drop('species', axis=1)

train_labels = train['species']
test_labels = test['species']

def input_function(features,labels,shuffle=False):
    labels = tf.keras.utils.to_categorical(labels).astype(np.float32)

    features = features.as_matrix().astype(np.float32)
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"layer_1_input": features},
        y=labels,
        shuffle=shuffle
    )
    return input_fn


def serving_input_receiver_fn():
    serialized_tf_example = tf.placeholder(dtype=tf.string,
                                           shape=[1],
                                           name='input_example_tensor')
    feature_spec = {'layer_1_input': tf.FixedLenFeature(shape=[4], dtype=tf.float32)}
    receiver_tensors = {'layer_1_input': serialized_tf_example}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(serialized_tf_example, receiver_tensors)


tf_model.train(input_fn=input_function(train_features, train_labels, True),steps=200)

print(tf_model.evaluate(input_function(test_features, test_labels,True)))

feature_spec = {'flower_features': tf.FixedLenFeature(shape=[4], dtype=np.float32)}

serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

tf_model.export_savedmodel("/Users/yazen/Desktop/mlprojects/Iris",serving_input_receiver_fn)