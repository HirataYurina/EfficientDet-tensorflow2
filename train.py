# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:train.py
# software: PyCharm
# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:train_keras.py
# software: PyCharm

import tensorflow.keras as keras
from net.neck_and_head.efficient_det import efficient_det, RESOLUTION_LIST
from config.configs import config
from dataset.get_dataset import DataGenerator
from core.loss import efficient_loss
import tensorflow as tf


if __name__ == '__main__':

    tf.executing_eagerly = False

    train_txt = config.TRAIN.TRAIN_TXT
    class_txt = config.TRAIN.CLASS_TXT
    batch_size1 = config.TRAIN.BATCH_SIZE1
    batch_size2 = config.TRAIN.BATCH_SIZE2
    model_id = config.TRAIN.MODEL_ID

    with open(train_txt) as f:
        train_anno = f.readlines()
    num_train = len(train_anno)
    with open(class_txt) as f:
        classes = f.readlines()
    num_classes = len(classes)

    train_steps_1 = num_train // batch_size1
    train_steps_2 = num_train // batch_size2

    retina_model, inputs = efficient_det(backbone=model_id,
                                         num_anchors=9,
                                         num_classes=num_classes)

    outputs = retina_model.outputs
    y_true = [keras.Input(shape=(None, 5)),
              keras.Input(shape=(None, num_classes + 1))]
    loss_input = [y_true, outputs]
    model_loss = keras.layers.Lambda(efficient_loss,
                                     output_shape=(1,),
                                     name='retina_loss')(loss_input)
    model = keras.Model([inputs, y_true], model_loss)

    # ####################################################
    # data generator
    data_gene_1 = DataGenerator(anno_lines=train_anno,
                                input_shape=(RESOLUTION_LIST[model_id], RESOLUTION_LIST[model_id]),
                                num_classes=num_classes,
                                batch_size=batch_size1)
    generate_data_1 = data_gene_1.data_generate()
    # ####################################################

    # optimizer1 = keras.optimizers.Adam(learning_rate=1e-04, clipnorm=0.001)
    optimizer1 = keras.optimizers.Adam(learning_rate=1e-04)
    # freeze the first 174 layers
    # training in stage 1
    num_freeze_layers = config.TRAIN.FREEZE
    for i in range(num_freeze_layers):
        retina_model.layers[i].trainable = False
    print('have frozen resnet model and start training')

    model.compile(optimizer=optimizer1,
                  loss={'retina_loss': lambda y_true, y_pred: y_pred})

    model.fit_generator(generator=generate_data_1,
                        steps_per_epoch=train_steps_1,
                        epochs=50)
