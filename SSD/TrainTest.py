#codeing = utf8
#只是为了测试下接口是否正确，并非Train函数
import tensorflow as tf
import SSD_300Net

imagepath = './1.jpg'
image = tf.read_file(imagepath)
image = tf.image.decode_jpeg(image, channels=3)
Input = tf.to_float(image)
Input = tf.expand_dims(Input, 0)

num_classes = 21
anchor_ratios = [[2, .5],
                 [2, .5, 3, 1. / 3],
                 [2, .5, 3, 1. / 3],
                 [2, .5, 3, 1. / 3],
                 [2, .5],
                 [2, .5]]

anchor_sizes = [(21., 45.),
                (45., 99.),
                (99., 153.),
                (153., 207.),
                (207., 261.),
                (261., 315.)]
feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11']

#先通过net获取每层的网络输出及预测值
predictions, localisations, logits, end_points = SSD_300Net.ssd_300Net(Input, num_classes, anchor_ratios, feat_layers)

img_shape = [300,300]
feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
anchor_steps=[8, 16, 32, 64, 100, 300]

#获取所有的锚点框
anchors_all = ssd_anchors_all_layers(img_shape, feat_shapes, anchor_sizes, anchor_ratios, anchor_steps)

#GT筛选
labels = 1
bboxes = [0,0,0.1,0.2]
g_labels, g_localizations, g_scores = ssd_bboxes_encode_layer(labels, bboxes, anchors_all, num_classes)

#loss
loss = NetLoss(logits, localisations, g_labels, g_localizations, g_scores)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(loss)