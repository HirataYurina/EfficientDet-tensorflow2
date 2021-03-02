# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:test_net.py
# software: PyCharm

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf

inputs = tf.random.uniform(shape=(8, 1))
inputs2 = tf.random.uniform(shape=(8, 1))
print(tf.stack([inputs, inputs2], axis=-3))