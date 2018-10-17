import tensorflow as tf
import tensorflow.contrib.slim as slim

def pad2d(Input, padSize):
    net = tf.pad(Input, [[0,0], [padSize[0],padSize[0]], [padSize[1],padSize[1]], [0,0]])
    return net

def tensor_shape(x):
    return x.get_shape().as_list()

def ssd_multibox_layer(inputs,
                       num_classes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    if normalization>0:
        print(normalization)
        # TODO: normalize,for block4
    num_anchors = (2 + len(ratios))
    num_location = num_anchors*4

    #predict location
    loc_pred = slim.conv2d(inputs, num_location, [3,3], activation_fn=None, scope='conv_loc')
    #use for what?
    #Get it: after conv_loc, feat shape is [H ,W , num_location]
    #But for easy division, reshape to [H,W,num_anchors, 4]
    loc_pred = tf.reshape(loc_pred, tensor_shape(loc_pred)[:-1] + [num_anchors, 4])

    #predict class
    cls_pred = slim.conv2d(inputs, num_anchors*num_classes, [3,3], activation_fn=None, scope='conv_cls')
    cls_pred = tf.reshape(cls_pred, tensor_shape(cls_pred)[:-1] + [num_anchors, num_classes])

    return cls_pred, loc_pred

def ssd_300Net(Input, num_classes,
               anchor_ratios,
               feat_layers,
               dropout_keep_prob=0.5,
               is_training=True,
               prediction_fn=slim.softmax):
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

    #block4 - out 38x38
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

    #block7 - out 19x19
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
            p,l = ssd_multibox_layer(end_points[layer], num_classes, anchor_ratios)
            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)


    return predictions, localisations, logits, end_points

