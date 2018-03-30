#-*- coding: UTF-8 -*-

"""
读取csv格式的文件
《TensorFlow入门与实战》第三章 TensorFlow基础
3.5 数据读取
"""

import tensorflow as tf

# 将文件列表名传入，shuffle=True表示读入的时候会乱序从各个文件读（打乱的是文件序列，不是文件内容的序列）
# num_epochs=2表示所有文件训练读取两次后结束
filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"], shuffle=True, num_epochs=2)

# 采用读文本的reader，按文本的行读
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# 默认值是0.1，这里也默认指定了要读入数据的数据类型是float
record_defaults = [[0.1],[0.1]]

# 获取读取的内容，采用decode_csv的方式来读取数据
# 默认是按照逗号来分割一行中的数据
v1, v2 = tf.decode_csv(value, record_defaults=record_defaults)

# 计算乘法
v_mul = tf.multiply(v1, v2)

# 快速初始化所有变量
init_op = tf.global_variables_initializer()
# 如果是采用TensorFlow的读取文件的方式
# 需要执行tf.local_variables_initializer
local_init_op = tf.local_variables_initializer()

# 创建会话
sess = tf.Session()

# 初始化变量
sess.run(init_op)
sess.run(local_init_op)

# 启动输入数据的队列，不执行这句代码，程序将一直处于等待数据的状态
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
try:
    while not coord.should_stop():
        value1, value2, mul_result = sess.run([v1, v2, v_mul])
        print("%f\t%f\t%f" % (value1, value2, mul_result))
except tf.errors.OutOfRangeError:
    print('Done trainging -- epoch limit reached')
finally:
    coord.request_stop()

# 等待线程结束
coord.join(threads)
sess.close()
