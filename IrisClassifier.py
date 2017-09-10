import pandas as pd

import tensorflow as tf
from helperFunctions import create_layer, cost_compute
input_features = 4
n_hidden_units1 = 10
n_hidden_units2 = 14
n_hidden_units3 = 12
n_hidden_units4 = 3

rate = .001

weights = dict(
            w1=tf.Variable(tf.random_normal([input_features, n_hidden_units1])),
            w2=tf.Variable(tf.random_normal([n_hidden_units1, n_hidden_units2])),
            w3=tf.Variable(tf.random_normal([n_hidden_units2, n_hidden_units3])),
            w4=tf.Variable(tf.random_normal([n_hidden_units3, n_hidden_units4]))
            )

biases = dict(
            b1=tf.Variable(tf.zeros([n_hidden_units1])),
            b2=tf.Variable(tf.zeros([n_hidden_units2])),
            b3=tf.Variable(tf.zeros([n_hidden_units3])),
            b4=tf.Variable(tf.zeros([n_hidden_units4]))
            )

train = pd.read_csv("/Users/yazen/Desktop/datasets/iris_training.csv")
test = pd.read_csv("/Users/yazen/Desktop/datasets/iris_test.csv")

train.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']
test.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']

train_features = train.drop('species', axis=1)
test_features = test.drop('species', axis=1)
train_features

test_labels = tf.one_hot(test['species'], depth=3)
train_labels = tf.one_hot(train['species'], depth=3)

test_labels
train_labels = tf.Session().run(train_labels)
test_labels = tf.Session().run(test_labels)


x = tf.placeholder("float32", [None, 4], name="asdfadsf")
y = tf.placeholder("float32", [None, 3], name="asdfasdf2")

layer = create_layer(x, weights['w1'], biases['b1'], tf.nn.relu)
layer = create_layer(layer, weights['w2'], biases['b2'], tf.nn.relu)
layer = create_layer(layer, weights['w3'], biases['b3'], tf.nn.relu)
Z4 = create_layer(layer, weights['w4'], biases['b4'])
cost = cost_compute(Z4, y)
optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iteration in range(1,500):
        _, c = sess.run([optimizer, cost], feed_dict={x: train_features, y: train_labels})
        print("Iteration " + str(iteration) + " cost: " + str(c))

    prediction = tf.argmax(Z4,1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Z4,1), tf.argmax(y,1)), "float"))
    print(sess.run(prediction, feed_dict={x: train_features, y: train_labels}))
    print(accuracy.eval({x: train_features, y: train_labels}))
    print(accuracy.eval({x: test_features, y: test_labels}))


