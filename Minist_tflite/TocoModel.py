# manually put back imported modules
import tensorflow as tf
import tempfile
import subprocess
tf.contrib.lite.tempfile = tempfile
tf.contrib.lite.subprocess = subprocess

path="minist.pb"        #pb文件位置和文件名
inputs=["input"]               #模型文件的输入节点名称
classes=["output"]            #模型文件的输出节点名称
converter = tf.contrib.lite.TocoConverter.from_frozen_graph(path,  inputs, classes)
tflite_model=converter.convert()
open("minist.tflite", "wb").write(tflite_model)







