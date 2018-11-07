#codeing = utf-8
import tensorflow as tf
import FRCNN

#read a test jpeg
imagepath = './1.jpg'
image = tf.read_file(imagepath)
image = tf.image.decode_jpeg(image, channels=3)
Input = tf.to_float(image)
Input = tf.expand_dims(Input, 0)

vgg_net = FRCNN.VGG16BaseNet(Input)

FRCNN.RPNNet(vgg_net)