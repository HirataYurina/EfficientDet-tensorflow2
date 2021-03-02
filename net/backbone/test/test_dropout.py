# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:test_dropout.py
# software: PyCharm

import tensorflow as tf
import tensorflow.keras.layers as layers


def test_drop(inputs):
    # we need to set training=True/False when we use dropout layer.
    batch_size = tf.shape(inputs)[0]
    results = layers.Dropout(rate=0.5, noise_shape=(batch_size, 1, 1))(inputs, training=True)
    return results


def test_stars(kernel_size, strides, padding):
    print(kernel_size)
    print(strides)
    print(padding)


class TestModel(tf.keras.Model):

    def __init__(self):
        super(TestModel, self).__init__()
        self.reshape = layers.Reshape((-1, 1))
        self.dense = layers.Dense(10)

    def call(self, inputs, training=None, mask=None):
        res = self.reshape(inputs)
        res = self.dense(res)
        return res


if __name__ == '__main__':

    # test dropout
    with tf.GradientTape() as tape:
        a = tf.ones(shape=(3, 3))
        res = test_drop(a)
    print(res)
    # And we can see that the result has been scaled by 2 when we use dropout.
    # Because drop_ratio=0.5, the expectation of output is scaled by 0.5.
    # tf.Tensor(
    #     [[[2. 2. 2.]
    #       [2. 2. 2.]
    #       [2. 2. 2.]]
    #      [[0. 0. 0.]
    #       [0. 0. 0.]
    #       [0. 0. 0.]]
    #      [[2. 2. 2.]
    #       [2. 2. 2.]
    #       [2. 2. 2.]]], shape = (3, 3, 3), dtype = float32)

    # test star
    options = {'kernel_size': 3, 'strides': 2, 'padding': 'same'}
    test_stars(**options)
