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

def condition(i, labels):
    r = tf.less(i, tf.shape(labels))
    return r[0]

#这个函数的主要用处：获取每一层锚点框和真实样本框的交并比，当交并比大于
#阈值的时候，就筛选出来，并通过后续遍历所有真实框，获取一个最大的交并比的锚点框的
#位置
def body(i, anchors_layer, labels, bboxes, num_classes, feat_labels, feat_scores,
                  feat_ymin, feat_xmin, feat_ymax, feat_xmax):
    label = labels[i]
    bbox = bboxes[i]
    jaccard_value = jaccard_with_anchors(bbox, anchors_layer)
    #其实是和上一次做比较
    mask = tf.greater(jaccard_value, feat_scores)
    mask = tf.logical_and(mask, feat_scores>-0.5)
    mask = tf.logical_and(mask, label<num_classes)
    imask = tf.cast(mask, tf.int64)
    fmask = tf.cast(mask, tf.float32)

    #如果交并比比之前大，则选出来，否则保持原值
    feat_labels = imask * label + (1-imask)*feat_labels
    feat_scores = imask*jaccard_value + (1-imask)*feat_scores

    feat_ymin = fmask*bbox[0] + (1-fmask)*feat_ymin
    feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
    feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
    feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

    return [i+1, feat_labels, feat_scores, feat_ymin, feat_xmin, feat_ymax, feat_xmax]

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

    shape = (yref.shape[0],  yref.shape[1], href.size)

    feat_labels = tf.zeros(shape, tf.int64)
    feat_scores = tf.zeros(shape, dtype)
    feat_ymin = tf.zeros(shape, dtype)
    feat_xmin = tf.zeros(shape, dtype)
    feat_ymax = tf.zeros(shape, dtype)
    feat_xmax = tf.zeros(shape, dtype)

    i = 0
    [i, feat_labels, feat_scores,
     feat_ymin, feat_xmin,
     feat_ymax, feat_xmax] = tf.while_loop(condition, body,
                                           [i, anchors_layer, labels,
                                            bboxes, num_classes,
                                            feat_labels, feat_scores,
                                            feat_ymin, feat_xmin,
                                            feat_ymax, feat_xmax])
    feat_cy = (feat_ymax + feat_ymin) / 2
    feat_cx = (feat_xmax + feat_xmin) / 2
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin

    feat_cy = (feat_cy - yref)/ href /prior_scaling[0]
    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    feat_h = tf.log(feat_h / href) / prior_scaling[2]
    feat_w = tf.log(feat_w / wref) / prior_scaling[3]
    feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    return feat_labels, feat_localizations, feat_scores


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

