import tensorflow as tf
import tensorflow.contrib.slim as slim
import LossCal

def pad2d(Input, padSize):
    net = tf.pad(Input, [[0,0], [padSize[0],padSize[0]], [padSize[1],padSize[1]], [0,0]])
    return net

def ssd_300Net(Input, feat_layers, dropout_keep_prob=0.5, is_training=True):
    end_points = {}
    #block1
    net = slim.repeat(2, Input, slim.conv2d, 64, [3,3], scope='conv1')
    end_points['block1'] = net
    net = slim.max_pool2d(net, [2,2], scope='pool1')

    #block2
    net = slim.repeat(2, net, slim.conv2d, 128, [3, 3], scope='conv2')
    end_points['block2'] = net
    net = slim.max_pool2d(net, [2, 2], scope='pool2')

    #block3
    net = slim.repeat(3, net, slim.conv2d, 256, [3, 3], scope='conv3')
    end_points['block3'] = net
    net = slim.max_pool2d(net, [2, 2], scope='pool3')

    #block4
    net = slim.repeat(3, net, slim.conv2d, 512, [3, 3], scope='conv4')
    end_points['block4'] = net
    net = slim.max_pool2d(net, [2, 2], scope='pool4')

    #block5
    net = slim.repeat(3, net, slim.conv2d, 512, [3, 3], scope='conv5')
    end_points['block5'] = net
    net = slim.max_pool2d(net, [3, 3], 1, scope='pool5')

    #block6
    net = slim.conv2d(net, 1024, [3,3], rate=6, scope='conv6')
    end_points['block6'] = net
    net = slim.dropout(net, keep_prob=dropout_keep_prob, training=is_training)

    #block7
    net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
    end_points['block7'] = net
    net = slim.dropout(net, keep_prob=dropout_keep_prob, training=is_training)

    #block8 -- out 10x10x512
    net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
    net = pad2d(net, [1,1])
    net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3',padding='VALID')
    end_points['block8'] = net

    #block9 -- out 5x5x256
    net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
    net = pad2d(net, [1,1])
    net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
    end_points['block9'] = net

    # block10 -- out 3x3x256
    net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
    net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
    end_points['block10'] = net

    # block11 -- out 1x1x256
    net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
    net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
    end_points['block11'] = net

    #predict
    predictions = []
    logits = []
    localisations = []
    for i,layer in enumerate(feat_layers):
        with tf.variable_scope(layer + '_box'):
            p,l = LossCal.lossSingleLay(layer)
            predictions.append(p)
            logits.append(l)


    return predictions, logits

