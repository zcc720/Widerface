# -*- coding: utf-8 -*-

from skimage import io
import shutil
import random
import os
import string

headstr = """\
<annotation>
    <folder>VOC2007</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''

image_path='E:/ZDHT/P81113/'

def all_path(filename):
    return os.path.join('E:/ZDHT/', filename)

def writexml(filename, head, bbxes, tail):
    filename = all_path("Annotations/%s.xml" % (filename.split('.')[0]))
    f = open(filename, "w")
    f.write(head)
    for bbx in bbxes:
        f.write(objstr % ('face', bbx[0], bbx[1], bbx[2], bbx[3]))
    f.write(tail)
    f.close()


def clear_dir():
    if shutil.os.path.exists(all_path('Annotations')):
        shutil.rmtree(all_path('Annotations'))
    if shutil.os.path.exists(all_path('ImageSets')):
        shutil.rmtree(all_path('ImageSets'))
    if shutil.os.path.exists(all_path('JPEGImages')):
        shutil.rmtree(all_path('JPEGImages'))

    shutil.os.mkdir(all_path('Annotations'))
    shutil.os.makedirs(all_path('ImageSets/Main'))
    shutil.os.mkdir(all_path('JPEGImages'))


def excute_datasets(datatype):
    f = open(all_path('ImageSets/Main/' + datatype + '.txt'), 'a')
    f_bbx = open(all_path('result.txt'), 'r')

    while True:
        filename = f_bbx.readline().strip('\n')
        print(filename)
        if not filename:
            break
        im = io.imread(all_path(image_path+filename))
        head = headstr % (filename, im.shape[1], im.shape[0], im.shape[2])
        nums = (f_bbx.readline()).strip('\n')

        bbxes = []
        for ind in range(int(nums)):
            bbx_info = (f_bbx.readline()).strip(' \n').split(' ')
            bbx = [int(bbx_info[i]) for i in range(len(bbx_info)-1)]
            bbxes.append(bbx)
            writexml(filename, head, bbxes, tailstr)
            shutil.copyfile(all_path(image_path+filename), all_path('JPEGImages/%s' % (filename)))
            f.write('%s\n' % (filename))
    f.close()
    f_bbx.close()



# 打乱样本
def shuffle_file(filename):
    f = open(filename, 'r+')
    lines = f.readlines()
    random.shuffle(lines)
    f.seek(0)
    f.truncate()
    f.writelines(lines)
    f.close()


if __name__ == '__main__':
    clear_dir()
    excute_datasets('train')


# # -*- coding: utf-8 -*-
# """
# Created on 18/11/12
#
# @author: zcc
# """
# from skimage import io
# import shutil
# import random
# import os
# import string
# headstr = """\
# <annotation>
#     <folder>VOC2007</folder>
#     <filename>%s</filename>
#     <source>
#         <database>My Database</database>
#         <annotation>PASCAL VOC2007</annotation>
#         <image>flickr</image>
#         <flickrid>NULL</flickrid>
#     </source>
#     <owner>
#         <flickrid>NULL</flickrid>
#         <name>company</name>
#     </owner>
#     <size>
#         <width>%d</width>
#         <height>%d</height>
#         <depth>%d</depth>
#     </size>
#     <segmented>0</segmented>
# """
# objstr = """\
#     <object>
#         <name>%s</name>
#         <pose>Unspecified</pose>
#         <truncated>0</truncated>
#         <difficult>0</difficult>
#         <bndbox>
#             <xmin>%d</xmin>
#             <ymin>%d</ymin>
#             <xmax>%d</xmax>
#             <ymax>%d</ymax>
#         </bndbox>
#     </object>
# """
#
# tailstr = '''\
# </annotation>
# '''
#
# def all_path(filename):
#     return os.path.join('./', filename)
#
# def writexml(name, head, bbxes, tail):
#     filename = all_path("Annotations/%s.xml" % (name.split('.')[0]))
#     f = open(filename, "w")
#     f.write(head)
#     f.write(objstr % ('trueface', bbxes[0], bbxes[1],bbxes[2],bbxes[3]))
#     f.write(tail)
#     f.close()
#
#
# def clear_dir():
#     if shutil.os.path.exists(all_path('Annotations')):
#         shutil.rmtree(all_path('Annotations'))
#     if shutil.os.path.exists(all_path('ImageSets')):
#         shutil.rmtree(all_path('ImageSets'))
#     if shutil.os.path.exists(all_path('JPEGImages')):
#         shutil.rmtree(all_path('JPEGImages'))
#
#     shutil.os.mkdir(all_path('Annotations'))
#     shutil.os.makedirs(all_path('ImageSets/Main'))
#     shutil.os.mkdir(all_path('JPEGImages'))
#
#
# def excute_datasets(datatype):
#     f = open(all_path('ImageSets/Main/' + datatype + '.txt'), 'a')
#     f_bbx = open(all_path('result_true.txt'), 'r')
#
#     for bbox in f_bbx.readlines():
#         bb_info = bbox.split()
#         filename_ = bb_info[0]
#         print(filename_)
#         box = [int(i) for i in bb_info[1:5]]
#         print(box)
#         im = io.imread('E:/ZDHT/raw/trueface/'+filename_)
#         head = headstr % (filename_, im.shape[1], im.shape[0], im.shape[2])
#         writexml(filename_, head, box, tailstr)
#         shutil.copyfile(all_path('E:/ZDHT/raw/trueface/'+filename_), all_path('JPEGImages/%s' % (filename_)))
#         f.write('%s\n' % (filename_))
#     f.close()
#     f_bbx.close()
#
#
# # 打乱样本
# def shuffle_file(filename):
#     f = open(filename, 'r+')
#     lines = f.readlines()
#     random.shuffle(lines)
#     f.seek(0)
#     f.truncate()
#     f.writelines(lines)
#     f.close()
#
#
# if __name__ == '__main__':
#     clear_dir()
#     excute_datasets('train')
#     # idx = excute_datasets(idx, 'val')