#-*- coding: UTF-8 -*-

"""
生成TFRecord格式文件，该文件作为ReadTFRecordData的输入
"""
import tensorflow as tf

def _int64_feature(value):
    # 将int类型的value构造城Feature对象
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

if __name__ == "__main__":
    filename0 = "file0.tfrecords"
    print("Writing", filename0)
    # 定义写入tfrecord文件的writer
    writer = tf.python_io.TFRecordWriter(filename0)
    for index in range(10):
        # 将数据填入example
        # 填入的数据的v1是0-9，v2是1-10
        example = tf.train.Example(features = tf.train.Features(feature = {
            'v1': _int64_feature(index),
            'v2': _int64_feature(index + 1)}))
        # 将example序列化后写入文件
        writer.write(example.SerializeToString())
    writer.close()

    filename1 = "file1.tfrecords"
    writer = tf.python_io.TFRecordWriter(filename1)
    for index in range(10, 20):
        # 将数据填入example
        # TensorFlow的Example类，是TensorFlow内部定义好的protocol buffer，
        # 先将数据填入Example类型的protocol buffer，然后序列化数据，最后将数据写入文件
        # 目前TensorFlow提供了两种protocol buffer
        # 一种是Example，就是本demo中使用的，另外一种是SequenceExample
        # SequenceExample主要用来写入不定长的序列数据
        example = tf.train.Example(features = tf.train.Features(feature = {
            'v1': _int64_feature(index),
            'v2': _int64_feature(index + 1)
        }))
        # 将example序列化后写入文件
        writer.write(example.SerializeToString())
        writer.close()

