RPN整体过程：
上图中的Proposal,实际上包含了几个重要过程：
1.经过预设框回归以及分类预测后，预设框回归值和锚点框进行反变换，获取预设框实际坐标值，类似SSD算法中的解码过程
2.RPN中有个在目标检测中比较通用的思想，就是NMS，非极大值抑制，作用主要是为了避免最后有多个预测框对应一个检测物体的情况，NMS的具体实现如下：
非极大值抑制的方法是：先假设有6个矩形框，根据分类器的类别分类概率做排序，假设从小到大属于车辆的概率 分别为A、B、C、D、E、F。
(1)从最大概率矩形框F开始，分别判断A~E与F的重叠度IOU是否大于某个设定的阈值;
(2)假设B、D与F的重叠度超过阈值，那么就扔掉B、D；并标记第一个矩形框F，是我们保留下来的。
(3)从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，重叠度大于一定的阈值，那么就扔掉；并标记E是我们保留下来的第二个矩形框。
就这样一直重复，找到所有被保留下来的矩形框。
tf中的做法如下：
indices = tf.image.non_max_suppression(proposals, scores, max_output_size=post_nms_topN, iou_threshold=nms_thresh)
boxes = tf.gather(proposals, indices)
boxes = tf.to_float(boxes)
scores = tf.gather(scores, indices)
scores = tf.reshape(scores, shape=(-1, 1))

3.ROI Pooling过程：
ROI Pooling从字面上理解，就是截取一段ROI出来，然后进行pool操作。但是截取的不是原图，而是我们FM base层最后得到的特征层，这样可以免去
我们大量重复的卷积操作。实现的关键函数 tf.image.crop_and_resize ， 这个是tensorflow原生API，可以对一个tensor进行特定的裁剪和缩放，裁剪
之后我们再进行一次pool

4.经过ROI Pooling后，经过几层全全连接层进行分类预测及边框回归，完成整个算法