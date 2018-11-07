#codeing =  utf-8
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

def VGG16BaseNet(Input):
    #block1
    net = slim.repeat(Input, 2, slim.conv2d, 64, [3,3], scope='conv1')
    net = slim.max_pool2d(net, [2,2], scope='pool1')

    #block2
    net = slim.repeat(net, 2, slim.conv2d, 128, [3,3], scope='conv2')
    net = slim.max_pool2d(net, [2,2], scope='pool2')

    #block3
    net = slim.repeat(net, 3, slim.conv2d, 256, [3,3], scope='conv3')
    net = slim.max_pool2d(net, [2,2], scope='pool3')

    #block4
    net = slim.repeat(net, 3, slim.conv2d, 512, [3,3], scope='conv4')
    net = slim.max_pool2d(net, [2,2], scope='pool4')

    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    return net

def ReshapeLayer(Input, dim, name):
    inputshape = tf.shape(Input)
    with tf.variable_scope(name) as scope:
        output = tf.reshape(Input, [inputshape[0], inputshape[1], 9, dim])
        return output

def RPNNet(Input, dtype = np.float32):
    rpn = slim.conv2d(Input,  512, [3,3], scope='rpn_conv/3x3')
    #cls, shape -> [N,H,W,18]
    net = slim.conv2d(rpn, 3*3*2, [1,1], padding='VALID', activation_fn=None, scope='rpn_cls_score')
    #shape  -> [N,H,W*9,2]

    rpn_cls_prob_reshape = ReshapeLayer(net, 2, 'rpn_cls_score_reshape')

    cls_prob_shape = tf.shape(rpn_cls_prob_reshape)

    rpn_cls_pred_softmax = tf.nn.softmax(tf.reshape(rpn_cls_prob_reshape, [ -1, 2]))

    rpn_cls_pred = tf.reshape(rpn_cls_pred_softmax, cls_prob_shape)

    rpn_cls_prob_reshape = ReshapeLayer(rpn_cls_pred, 9*2, 'rpn_cls_prob_reshape')

    #position
    net_p = slim.conv2d(rpn, 3*3*4, [1,1], padding='VALID', activation_fn=None, scope='rpn_bbox_pred')

    return rpn_cls_prob_reshape, net_p