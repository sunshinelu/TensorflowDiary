#coding=utf-8
import tensorflow as tf

"""
使用placeholder填充方式读取数据
《TensorFlow入门与实战》第三章 TensorFlow基础 
3.5 数据读取
"""

v1 = tf.placeholder(tf.float32)
v2 = tf.placeholder(tf.float32)
v_mul = tf.multiply(v1,v2)

with tf.Session() as sess:
  while True:
    # 接受命令行的输入数据来填充
    # 当然，这里也可以自由地通过任何方式来填充
    value1 = input("value1: ")
    value2 = input("value2: ")
    # 将输入的数据通过feed_dict的参数传给会话的run函数
    mul_result = sess.run(v_mul, feed_dict={v1: value1, v2: value2})
    # 打印乘法结果
    print(mul_result)
