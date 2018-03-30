#-*- coding: UTF-8 -*-
"""
生成二进制文件file0.bin和file1.bin
生成对文件是ReadBinaryData的输入
"""

import struct
import codecs

filename0 = "file0.bin"

with codecs.open(filename0,'wb') as fw:
    for i in range(10):
        str1 = struct.pack('f',i)
        fw.write(str1)

filename1 = "file1.bin"

with codecs.open(filename1,'wb') as fw:
    for i in range(10,20):
        str1 = struct.pack('f',i)
        fw.write(str1)