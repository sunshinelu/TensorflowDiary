#-*- coding: UTF-8 -*-

import tensorflow as tf

x = tf.constant(1.0, name="input")
w = tf.Variable(0.5, name="weight")
b = tf.Variable(0.1, name="bias")
y = tf.add(tf.multiply(x, w, name="mul_op"), b, name="add_op")

# 设置写入的文件夹
summary_writer = tf.summary.FileWriter('./calc_graph')

# 获取默认的图
graph = tf.get_default_graph()
summary_writer.add_graph(graph)

# 将图结构写入文件
summary_writer.flush()
