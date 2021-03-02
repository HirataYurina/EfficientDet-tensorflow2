# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:test_model_in_model.py
# software: PyCharm

import tensorflow.keras as keras
import tensorflow as tf


class Prediction(keras.Model):
    """Layers have params don't be put in initial function"""
    def __init__(self):
        super(Prediction, self).__init__()

    def call(self, inputs, training=None, mask=None):
        x = keras.layers.Dense(20)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('sigmoid')(x)
        return x


class Prediction2(keras.Model):
    """Layers have params are put in initial function"""
    def __init__(self):
        super(Prediction2, self).__init__()
        self.dense = keras.layers.Dense(20)
        self.bn = keras.layers.BatchNormalization()
        self.sigmoid = keras.layers.Activation('sigmoid')

    def call(self, inputs, training=None, mask=None):
        x = self.dense(inputs)
        x = self.bn(x)
        x = self.sigmoid(x)
        x = keras.layers.Reshape((-1, 20))(x)

        return x


if __name__ == '__main__':
    # test
    model1 = Prediction()
    img_inputs = keras.Input(shape=(224, 224, 3))
    y = model1(img_inputs)
    model2 = keras.Model(img_inputs, y)
    # print(model2.trainable_variables)
    # model2.summary()

    # we test model in model in tf2.0
    model3 = Prediction2()
    # model3.summary()
    img_inputs = keras.Input(shape=(224, 224, 3))
    y = model3(img_inputs)
    model4 = keras.Model(img_inputs, y)
    # print(model4.trainable_variables)
    # model4.summary()

    # we test back propagation in tf.GradientTape
    # if we don't put layer has params in the init method,
    # we can't get gradients.
    image = tf.ones((1, 224, 224, 3))

    with tf.GradientTape() as tape:
        pred = model2(image)
    gradients = tape.gradient(pred, model2.trainable_variables)
    print(gradients)
    print(pred.shape)  # (1, 50176, 20)
    print(pred)
