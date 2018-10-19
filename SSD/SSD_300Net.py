#codeing = utf8
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math

def ssd_anchor_one_layer(image_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    y,x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]

    y = (y.astype(dtype) + offset) * step / image_shape[0]
    x = (x.astype(dtype) + offset) * step / image_shape[1]
    #for what?
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    num_anchors = 2 + len(ratios)
    w = np.zeros(num_anchors, dtype=dtype)
    h = np.zeros(num_anchors, dtype=dtype)

    w[0] = sizes[0] / image_shape[1]
    h[0] = sizes[0] / image_shape[0]

    w[1] = math.sqrt(sizes[0]*sizes[1]) / image_shape[1]
    h[1] = math.sqrt(sizes[0] * sizes[1]) / image_shape[0]

    for i,ratio in enumerate(ratios):
        w[i+2] = sizes[0] / image_shape[1] / math.sqrt(ratio)
        h[i + 2] = sizes[0] / image_shape[0] * math.sqrt(ratio)
    return y,x,h,w

def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    layers_anchors = []
    for i,lay_shape in enumerate(layers_shape):
        anchor_box = ssd_anchor_one_layer(img_shape, lay_shape, anchor_sizes, anchor_ratios, anchor_steps, offset, dtype)
        layers_anchors.append(anchor_box)
    return  layers_anchors

#实际上就是求个交并比
def jaccard_with_anchors(bbox, anchors_layer):
    #center
    yref = anchors_layer[0]
    xref = anchors_layer[1]
    href = anchors_layer[2]
    wref = anchors_layer[3]
    #left top
    ymin = yref - href/2
    xmin = xref - wref/2
    #right bottom
    ymax = yref + href/2
    xmax = xref + wref / 2

    vol_anchors = href*wref

    y_jmin = tf.maximum(ymin, bbox[0])
    x_jmin = tf.maximum(xmin, bbox[1])

    y_jmax = tf.minimum(ymax, bbox[2])
    x_jmax = tf.minimum(xmax, bbox[3])

    h = tf.maximum(y_jmax - y_jmin, 0)
    w = tf.maximum(x_jmax - x_jmin, 0)
    #交集面积
    vol_j = h*w
    #并集面积，这里使用了一点小技巧，并集面积 = 2部分面积和 - 交集
    vol_union =vol_anchors +(bbox[2]-bbox[0])*(bbox[3]-bbox[1]) - vol_j
    return tf.div(vol_j, vol_union)

def intersection_with_anchors(bbox, anchors_layer):
    #center
    yref = anchors_layer[0]
    xref = anchors_layer[1]
    href = anchors_layer[2]
    wref = anchors_layer[3]
    #left top
    ymin = yref - href/2
    xmin = xref - wref/2
    #right bottom
    ymax = yref + href/2
    xmax = xref + wref / 2

    vol_anchors = href*wref

    y_jmin = tf.maximum(ymin, bbox[0])
    x_jmin = tf.maximum(xmin, bbox[1])

    y_jmax = tf.minimum(ymax, bbox[2])
    x_jmax = tf.minimum(xmax, bbox[3])

    h = tf.maximum(y_jmax - y_jmin, 0)
    w = tf.maximum(x_jmax - x_jmin, 0)
    #交集面积
    vol_j = h*w
    return tf.div(vol_j, vol_anchors)

def ssd_bboxes_encode_layer(labels,
                            bboxes,
                            anchors_layer,
                            num_classes,
                            no_annotation_label,
                            ignore_threshold=0.5,
                            prior_scaling=[0.1, 0.1, 0.2, 0.2],
                            dtype=tf.float32):
    #center
    yref = anchors_layer[0]
    xref = anchors_layer[1]
    href = anchors_layer[2]
    wref = anchors_layer[3]
    #left top
    ymin = yref - href/2
    xmin = xref - wref/2
    #right bottom
    ymax = yref + href/2
    xmax = xref + wref / 2

    vol_anchors = href*wref




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

