import pandas as pd
import tensorflow as tf
from helperFunctions import create_layer, cost_compute
import keras as ks

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

train = pd.read_csv("/Users/yazen/Desktop/datasets/iris/iris_training.csv")
test = pd.read_csv("/Users/yazen/Desktop/datasets/iris/iris_test.csv")

train.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']
test.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']

train_features = train.drop('species', axis=1)
feat = train_features.columns

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

cost = cost_compute(Z4, y,weights)
optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost)

def serving_input_receiver_fn():
    serialized_tf_example = tf.placeholder(dtype=tf.string,
                                           shape=[1],
                                           name='input_example_tensor')
    feature_spec = {'layer_1_input': tf.FixedLenFeature(shape=[4], dtype=tf.float32)}
    receiver_tensors = {'layer_1_input': serialized_tf_example}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(serialized_tf_example, receiver_tensors)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iteration in range(1,2000):
        _, c = sess.run([optimizer, cost], feed_dict={x: train_features, y: train_labels})
        print("Iteration " + str(iteration) + " cost: " + str(c))

    prediction = tf.argmax(Z4,1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Z4,1), tf.argmax(y,1)), "float"))
    print(sess.run(prediction, feed_dict={x: train_features, y: train_labels}))
    print(accuracy.eval({x: train_features, y: train_labels}))
    print(accuracy.eval({x: test_features, y: test_labels}))

    tf.estimator.Estimator.export_savedmodel(export_dir_base="/Users/yazen/Desktop/mlprojects/Iris",serving_input_receiver_fn= serving_input_receiver_fn)


    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = accuracy.Estimator(
        model_fn=accuracy,
        params={
            'feature_columns': feat,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 3,
        })



# model = ks.models.Sequential()
#
# model.add(ks.layers.Dense(10, input_shape=(input_features,), activation='relu', kernel_regularizer=ks.regularizers.l2()))
# model.add(ks.layers.Dense(14, activation='relu', kernel_regularizer=ks.regularizers.l2()))
# model.add(ks.layers.Dense(12, activation='relu', kernel_regularizer=ks.regularizers.l2()))
# model.add(ks.layers.Dense(3, activation='softmax'))
#
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_features.as_matrix(), train_labels, verbose=1, epochs=200)
# print(model.evaluate(test_features.as_matrix(), test_labels))
