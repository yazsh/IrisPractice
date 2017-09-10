import tensorflow as tf


def create_layer(previous_layer, weight, bias, activation_function=None):
    z = tf.add(tf.matmul(previous_layer, weight),bias)
    if activation_function is None:
        return z
    a = activation_function(z)
    return a


def cost_compute(prediction, correct_values):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = correct_values))