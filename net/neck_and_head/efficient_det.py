# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:efficient_det.py
# software: PyCharm

import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPool2D, UpSampling2D, BatchNormalization
import tensorflow as tf
from net.backbone.efficient_net import efficient_net_b0, efficient_net_b1, efficient_net_b2, efficient_net_b3
from net.backbone.efficient_net import efficient_net_b4, efficient_net_b5, efficient_net_b6, efficient_net_b7
from net.backbone.efficient_net import CONV_KERNEL_INITIALIZER
from core.bias_initializer import BiasInitializer
import tensorflow.keras.layers as layers

# BN global parameters
MOMENTUM = 0.997
EPSILON = 1e-4
# the params of efficient net
RESOLUTION_LIST = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
WIDTH_LIST = [64, 88, 112, 160, 224, 288, 384, 384]
DEPTH_BIFPN_LIST = [3, 4, 5, 6, 7, 7, 8, 8]
DEPTH_CLASS_LIST = [3, 3, 3, 4, 4, 4, 5, 5]


class Swish(keras.layers.Layer):
    """
        swish activation
        y = x * sigmoid(x)
    """
    def __init__(self):
        super(Swish, self).__init__()

    def call(self, inputs, **kwargs):
        results = layers.multiply([inputs, tf.keras.activations.sigmoid(inputs)])
        return results


class WeightAdd(keras.layers.Layer):
    """Weighted feature fusion"""
    def __init__(self, epsilon=1e-4, **kwargs):
        super(WeightAdd, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        num_weights = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                 shape=(num_weights,),
                                 dtype=tf.float32,
                                 trainable=True,
                                 initializer=keras.initializers.constant(1 / num_weights))

    def call(self, inputs, **kwargs):
        # 3.3. Weighted Feature Fusion
        # fast fusion approach has very similar learning behavior
        # and accuracy as the softmax-based fusion, but runs up to
        # 30% faster on GPUs
        w = keras.activations.relu(self.w)
        w = w / (tf.reduce_sum(w) + self.epsilon)
        results = [inputs[i] * w[i] for i in range(len(inputs))]
        results = tf.reduce_sum(results, axis=0)
        return results


def bifpn(features, out_channels, ids, training=True):
    # The first Bifpn layer
    if ids == 0:
        _, _, c3, c4, c5 = features
        p3 = Conv2D(out_channels, 1, 1, padding='same')(c3)
        p3 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(p3, training=training)

        p4 = Conv2D(out_channels, 1, 1, padding='same')(c4)
        p4 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(p4, training=training)

        p5 = Conv2D(out_channels, 1, 1, padding='same')(c5)
        p5 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(p5, training=training)

        p6 = Conv2D(out_channels, 1, 1, padding='same')(c5)
        p6 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(p6, training=training)
        p6 = MaxPool2D(3, 2, padding='same')(p6)

        p7 = MaxPool2D(3, 2, padding='same')(p6)
        p7_up = UpSampling2D(2)(p7)

        p6_middle = WeightAdd()([p6, p7_up])
        p6_middle = Swish()(p6_middle)
        p6_middle = SeparableConv2D(out_channels, 3, 1, padding='same')(p6_middle)
        p6_middle = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(p6_middle, training=training)
        p6_up = UpSampling2D(2)(p6_middle)

        p5_middle = WeightAdd()([p5, p6_up])
        p5_middle = Swish()(p5_middle)
        p5_middle = SeparableConv2D(out_channels, 3, padding='same')(p5_middle)
        p5_middle = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(p5_middle, training=training)
        p5_up = UpSampling2D(2)(p5_middle)

        p4_middle = WeightAdd()([p4, p5_up])
        p4_middle = Swish()(p4_middle)
        p4_middle = SeparableConv2D(out_channels, 3, padding='same')(p4_middle)
        p4_middle = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(p4_middle, training=training)
        p4_up = UpSampling2D(2)(p4_middle)

        p3_out = WeightAdd()([p3, p4_up])
        p3_out = Swish()(p3_out)
        p3_out = SeparableConv2D(out_channels, 3, padding='same')(p3_out)
        p3_out = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(p3_out, training=training)
        p3_down = MaxPool2D(3, strides=2, padding='same')(p3_out)

        # path aggregation
        p4_out = WeightAdd()([p4, p4_middle, p3_down])
        p4_out = Swish()(p4_out)
        p4_out = SeparableConv2D(out_channels, kernel_size=3, strides=1, padding='same')(p4_out)
        p4_out = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(p4_out, training=training)
        p4_down = MaxPool2D(pool_size=3, strides=2, padding='same')(p4_out)

        p5_out = WeightAdd()([p5, p5_middle, p4_down])
        p5_out = Swish()(p5_out)
        p5_out = SeparableConv2D(out_channels, 3, 1, padding='same')(p5_out)
        p5_out = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(p5_out, training=training)
        p5_down = MaxPool2D(3, strides=2, padding='same')(p5_out)

        p6_out = WeightAdd()([p6, p6_middle, p5_down])
        p6_out = Swish()(p6_out)
        p6_out = SeparableConv2D(out_channels, kernel_size=3, strides=1, padding='same')(p6_out)
        p6_out = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(p6_out, training=training)
        p6_down = MaxPool2D(pool_size=3, strides=2, padding='same')(p6_out)

        p7_out = WeightAdd()([p7, p6_down])
        p7_out = Swish()(p7_out)
        p7_out = SeparableConv2D(out_channels, kernel_size=3, strides=1, padding='same')(p7_out)
        p7_out = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(p7_out, training=training)

    # Not the first Bifpn layer
    else:
        p3, p4, p5, p6, p7 = features

        p7_up = UpSampling2D(2)(p7)

        p6_middle = WeightAdd()([p6, p7_up])
        p6_middle = Swish()(p6_middle)
        p6_middle = SeparableConv2D(out_channels, 3, 1, padding='same')(p6_middle)
        p6_middle = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(p6_middle, training=training)
        p6_up = UpSampling2D(2)(p6_middle)

        p5_middle = WeightAdd()([p5, p6_up])
        p5_middle = Swish()(p5_middle)
        p5_middle = SeparableConv2D(out_channels, 3, padding='same')(p5_middle)
        p5_middle = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(p5_middle, training=training)
        p5_up = UpSampling2D(2)(p5_middle)

        p4_middle = WeightAdd()([p4, p5_up])
        p4_middle = Swish()(p4_middle)
        p4_middle = SeparableConv2D(out_channels, 3, padding='same')(p4_middle)
        p4_middle = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(p4_middle, training=training)
        p4_up = UpSampling2D(2)(p4_middle)

        p3_out = WeightAdd()([p3, p4_up])
        p3_out = Swish()(p3_out)
        p3_out = SeparableConv2D(out_channels, 3, padding='same')(p3_out)
        p3_out = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(p3_out, training=training)
        p3_down = MaxPool2D(3, strides=2, padding='same')(p3_out)

        # path aggregation
        p4_out = WeightAdd()([p4, p4_middle, p3_down])
        p4_out = Swish()(p4_out)
        p4_out = SeparableConv2D(out_channels, kernel_size=3, strides=1, padding='same')(p4_out)
        p4_out = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(p4_out, training=training)
        p4_down = MaxPool2D(pool_size=3, strides=2, padding='same')(p4_out)

        p5_out = WeightAdd()([p5, p5_middle, p4_down])
        p5_out = Swish()(p5_out)
        p5_out = SeparableConv2D(out_channels, 3, 1, padding='same')(p5_out)
        p5_out = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(p5_out, training=training)
        p5_down = MaxPool2D(3, strides=2, padding='same')(p5_out)

        p6_out = WeightAdd()([p6, p6_middle, p5_down])
        p6_out = Swish()(p6_out)
        p6_out = SeparableConv2D(out_channels, kernel_size=3, strides=1, padding='same')(p6_out)
        p6_out = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(p6_out, training=training)
        p6_down = MaxPool2D(pool_size=3, strides=2, padding='same')(p6_out)

        p7_out = WeightAdd()([p7, p6_down])
        p7_out = Swish()(p7_out)
        p7_out = SeparableConv2D(out_channels, kernel_size=3, strides=1, padding='same')(p7_out)
        p7_out = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(p7_out, training=training)

    return p3_out, p4_out, p5_out, p6_out, p7_out


class ClassPrediction(keras.Model):
    """We use SeparableConv2D in class prediction.
       The shape of result is (batch_size, h, w, num_anchors*num_classes)
    """
    def __init__(self, width, depth, num_classes, num_anchors, **kwargs):
        """

        Args:
            width:       this width is same as bifpn's width
            depth:       the number of layers
            num_classes: number of classes in our mission
            num_anchors: default of num_anchors is 3*3
        """
        super(ClassPrediction, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # the shared params of separable conv
        separable_conv_params = {'kernel_size': 3, 'strides': 1, 'padding': 'same'}
        # the shared params of BN
        bn_params = {'momentum': 0.997, 'epsilon': 0.0001}

        # #######################################################
        # Conv2D layers are spared by different level feature
        # So, there are depth BN layers
        # #######################################################
        self.separable_conv = \
            [SeparableConv2D(filters=width, **separable_conv_params,
                             kernel_initializer=keras.initializers.VarianceScaling(**CONV_KERNEL_INITIALIZER))
             for _ in range(depth)]
        # #######################################################
        # BN layers are not spared by different level feature
        # So, there are 5*depth BN layers
        # #######################################################
        self.bn = [[BatchNormalization(**bn_params) for _ in range(3, 8)] for _ in range(depth)]
        self.swish = Swish()

        # class head
        # TODO: use prior probability bias initializer in focal loss [have done]
        self.head = SeparableConv2D(filters=self.num_anchors * self.num_classes,
                                    **separable_conv_params,
                                    kernel_initializer=keras.initializers.Constant(value=0),
                                    bias_initializer=BiasInitializer())
        self.reshape = keras.layers.Reshape(target_shape=(-1, num_classes))
        self.sigmoid = keras.layers.Activation(tf.nn.sigmoid)
        self.level = 0

    def call(self, inputs, training=None, mask=None):

        x = inputs

        for i in range(self.depth):
            x = self.separable_conv[i](x)
            x = self.bn[i][self.level % 5](x)
            x = self.swish(x)
        x = self.head(x)
        x = self.sigmoid(x)
        x = self.reshape(x)

        # level increasing
        self.level += 1

        return x


class BoxPrediction(keras.Model):
    """
        (h, w, num_anchors, 4)
    """
    def __init__(self, width, depth, num_anchors, **kwargs):
        super(BoxPrediction, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors

        # the shared params of separable conv2d
        separable_conv_params = {'kernel_size': 3, 'strides': 1, 'padding': 'same'}
        # the shared params of BN
        bn_params = {'momentum': 0.997, 'epsilon': 0.0001}

        self.conv = \
            [SeparableConv2D(width, **separable_conv_params,
             kernel_initializer=keras.initializers.VarianceScaling(**CONV_KERNEL_INITIALIZER)) for _ in range(depth)]
        self.bn = [[BatchNormalization(**bn_params) for _ in range(3, 8)] for _ in range(depth)]
        self.head = SeparableConv2D(num_anchors * 4, **separable_conv_params)
        self.swish = Swish()
        self.reshape = keras.layers.Reshape((-1, 4))

        self.level = 0

    def call(self, inputs, training=None, mask=None):
        x = inputs

        for i in range(self.depth):
            x = self.conv[i](x)
            x = self.bn[i][self.level % 5](x)
            x = self.swish(x)
        x = self.head(x)
        x = self.reshape(x)

        self.level += 1

        return x


def efficient_det(backbone,
                  num_anchors,
                  num_classes):
    """Efficient det model
    Args:
        backbone:    [int] the id of efficient net
        num_anchors: [int]
        num_classes: [int]

    Returns: model

    """
    assert backbone in range(8)
    img_shape = RESOLUTION_LIST[backbone]
    width = WIDTH_LIST[backbone]
    depth_bifpn = DEPTH_BIFPN_LIST[backbone]
    depth_class = DEPTH_CLASS_LIST[backbone]

    img_inputs = keras.Input(shape=(img_shape, img_shape, 3))

    features, _ = efficient_net_b0(img_inputs)

    # choose backbone
    if backbone == 1:
        features, _ = efficient_net_b1(img_inputs)
    elif backbone == 2:
        features, _ = efficient_net_b2(img_inputs)
    elif backbone == 3:
        features, _ = efficient_net_b3(img_inputs)
    elif backbone == 4:
        features, _ = efficient_net_b4(img_inputs)
    elif backbone == 5:
        features, _ = efficient_net_b5(img_inputs)
    elif backbone == 6:
        features, _ = efficient_net_b6(img_inputs)
    elif backbone == 7:
        features, _ = efficient_net_b7(img_inputs)

    # use Bifpn to predict
    for i in range(depth_bifpn):
        p3_out, p4_out, p5_out, p6_out, p7_out = bifpn(features, out_channels=width, ids=i)
        features = [p3_out, p4_out, p5_out, p6_out, p7_out]

    # start class prediction and box prediction
    class_pred = ClassPrediction(width, depth_class, num_classes, num_anchors)
    class_prob = [class_pred(feature) for feature in features]

    box_pred = BoxPrediction(width, depth_class, num_anchors)
    box_offset = [box_pred(feature) for feature in features]

    class_prob = keras.layers.Concatenate(axis=1)(class_prob)
    box_offset = keras.layers.Concatenate(axis=1)(box_offset)

    eff_det_model = keras.Model(img_inputs, [box_offset, class_prob])

    return eff_det_model, img_inputs


if __name__ == '__main__':
    # inputs = keras.Input(shape=(224, 224, 3))
    # res = Swish()(inputs)
    # model = keras.Model(inputs, res)
    # model(inputs, training=False)
    # model.summary()

    # num_anchors=9
    # num_classes=20
    # we use binary cross entropy for class prediction
    # yolo3: we simply use [independent] logistic classifier for every class
    efficient_det1 = efficient_det(1, 9, 20)
    efficient_det1.summary()

    dummy = tf.random.normal(shape=(1, 640, 640, 3))
    efficient_det1(dummy)
