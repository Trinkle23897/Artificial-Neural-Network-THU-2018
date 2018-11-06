# -*- coding: utf-8 -*-

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

class Model:
    def __init__(self,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.y_ = tf.placeholder(tf.int32, [None])
        self.init = tf.contrib.layers.xavier_initializer()

        # TODO:  fill the blank of the arguments
        self.loss, self.pred, self.acc = self.forward(is_train=True, reuse=False)
        self.loss_val, self.pred_val, self.acc_val = self.forward(is_train=False, reuse=True)

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        
        # TODO:  maybe you need to update the parameter of batch_normalization?
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                                var_list=self.params)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def forward(self, is_train, reuse=None):
    
        with tf.variable_scope("model", reuse=reuse):
            # TODO: implement input -- Conv -- BN -- ReLU -- Dropout -- MaxPool -- Conv -- BN -- ReLU -- Dropout -- MaxPool -- Linear -- loss
            #        the 10-class prediction output is named as "logits"
            # Your Conv Layer
            conv1 = tf.layers.conv2d(self.x_, filters=FLAGS.h1, kernel_size=[3, 3], strides=[1, 1], padding="SAME", kernel_initializer=self.init)
            # Your BN Layer: use batch_normalization_layer function
            bn1 = batch_normalization_layer(conv1, is_train)
            # Your Relu Layer
            relu1 = tf.nn.relu(bn1)
            # Your Dropout Layer: use dropout_layer function
            drop1 = dropout_layer(relu1, FLAGS.drop_rate, is_train)
            # Your MaxPool
            maxpool1 = tf.layers.max_pooling2d(drop1, pool_size=[2, 2], strides=2)
            # Your Conv Layer
            conv2 = tf.layers.conv2d(maxpool1, filters=FLAGS.h2, kernel_size=[3, 3], strides=[1, 1], padding="SAME", kernel_initializer=self.init)
            # Your BN Layer: use batch_normalization_layer function
            bn2 = batch_normalization_layer(conv2, is_train)
            # Your Relu Layer
            relu2 = tf.nn.relu(bn2)
            # Your Dropout Layer: use dropout_layer function
            drop2 = dropout_layer(relu2, FLAGS.drop_rate, is_train)
            # Your MaxPool
            maxpool2 = tf.layers.max_pooling2d(drop2, pool_size=[2, 2], strides=2)
            # Your Linear Layer
            reshape2 = tf.reshape(maxpool2, [-1, 7*7*FLAGS.h2])
            # logits = tf.Variable(tf.constant(0.0, shape=[100, 10]))  # deleted this line after you implement above layers
            logits = tf.layers.dense(reshape2, units=10)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        pred = tf.argmax(logits, 1)  # Calculate the prediction result
        correct_pred = tf.equal(tf.cast(pred, tf.int32), self.y_)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch
        
        return loss, pred, acc

def batch_normalization_layer(incoming, is_train=True):
    # TODO: implement the batch normalization function and applied it on fully-connected layers
    # NOTE:  If isTrain is True, you should return calculate mu and sigma by mini-batch
    #       If isTrain is False, you must estimate mu and sigma from training data
    return tf.layers.batch_normalization(incoming, training=is_train)
    
def dropout_layer(incoming, drop_rate, is_train=True):
    # TODO: implement the dropout function and applied it on fully-connected layers
    # Note: When drop_rate=0, it means drop no values
    #       If isTrain is True, you should randomly drop some values, and scale the others by 1 / (1 - drop_rate)
    #       If isTrain is False, remain all values not changed
    return tf.nn.dropout(incoming, drop_rate) if is_train else incoming
