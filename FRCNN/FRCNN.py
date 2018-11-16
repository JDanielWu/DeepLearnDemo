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

def bbox_transform_inv_tf(boxes, deltas):
    boxes = tf.cast(boxes, deltas.dtype)
    widths = tf.subtract(boxes[:, 2], boxes[:, 0]) + 1.0
    heights = tf.subtract(boxes[:, 3], boxes[:, 1]) + 1.0
    ctr_x = tf.add(boxes[:, 0], widths * 0.5)
    ctr_y = tf.add(boxes[:, 1], heights * 0.5)

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = tf.add(tf.multiply(dx, widths), ctr_x)
    pred_ctr_y = tf.add(tf.multiply(dy, heights), ctr_y)
    pred_w = tf.multiply(tf.exp(dw), widths)
    pred_h = tf.multiply(tf.exp(dh), heights)

    pred_boxes0 = tf.subtract(pred_ctr_x, pred_w * 0.5)
    pred_boxes1 = tf.subtract(pred_ctr_y, pred_h * 0.5)
    pred_boxes2 = tf.add(pred_ctr_x, pred_w * 0.5)
    pred_boxes3 = tf.add(pred_ctr_y, pred_h * 0.5)

    return tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
     """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                        y_ctr - 0.5 * (hs - 1),
                        x_ctr + 0.5 * (ws - 1),
                        y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                       for i in range(ratio_anchors.shape[0])])
    return anchors

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

    #这些只是为了展示FRCNN的整体流程，可能具体调用的时候会有数据格式问题
    anchors = generate_anchors()
    proposals_bbox = bbox_transform_inv_tf(anchors ,net_p)

    indices = tf.image.non_max_suppression(proposals_bbox, rpn_cls_prob_reshape, max_output_size=5, iou_threshold=0.5)
    boxes = tf.gather(proposals_bbox, indices)
    boxes = tf.to_float(boxes)
    scores = tf.gather(rpn_cls_prob_reshape, indices)
    scores = tf.reshape(scores, shape=(-1, 1))

    return boxes, scores

def ROIPooling(FMNet, rois):
    bottom_shape = tf.shape(FMNet)
    #只是为了理解过程
    height = 224
    width = 224
    x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
    y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
    x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
    y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
    # Won't be back-propagated to rois anyway, but to save time
    bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
    pre_pool_size = 20 * 2
    crops = tf.image.crop_and_resize(FMNet, bboxes, tf.to_int32(0), [pre_pool_size, pre_pool_size],
                                     name="crops")
    ROIPoolNet = slim.max_pool2d(crops, [2, 2], padding='SAME')
    return ROIPoolNet
