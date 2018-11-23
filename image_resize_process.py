# import os
# import random
# random.seed(10)
# import shutil
# import cv2
#
# def reconstruction_image(path):
#     file_list=os.listdir(path)
#     slice = random.sample(file_list, 2000)
#     return slice
#
#
# if __name__ == '__main__':
#
#     path='E:/code/Widerface/widerface/JPEGImages/'
#     xml_path='E:/code/Widerface/widerface/Annotations/'
#     image_savepath='E:/code/Widerface/widerface/process_img/'
#     xml_savepath='E:/code/Widerface/widerface/process_xml/'
#
#     if shutil.os.path.exists(image_savepath):
#         shutil.rmtree(image_savepath)
#
#     if shutil.os.path.exists(xml_savepath):
#         shutil.rmtree(xml_savepath)
#
#
#     shutil.os.mkdir(xml_savepath)
#     shutil.os.mkdir(image_savepath)
#
#     slice=reconstruction_image(path)
#     for i in slice:
#         img = cv2.imread(path+i)
#         h,w,c=img.shape    #获取图片大小（长，宽，通道数）
#         # cv2.imshow('image', img)
#         img2 = cv2.resize(img, (int(w * 0.5), int(h * 0.5)))
#         tempimg = cv2.resize(img2, (w, h ))
#
#         cv2.imwrite(image_savepath+'1_'+i, tempimg)
#         shutil.copyfile(xml_path+i.split('.')[0]+'.xml',xml_savepath+'1_'+i.split('.')[0]+'.xml')

# -*- coding: UTF-8 -*-
import os, h5py, cv2, sys, shutil
import numpy as np
from xml.dom.minidom import Document
from PIL import Image

rootdir = "./widerface"

convert2vocformat = True

# 最小取20大小的脸，并且补齐
minsize2select = 20
def all_path(filename):
    return os.path.join(save_path, filename)

datasetprefix = "./widerface"  #
save_path='E:/wider_rotate/up/'
imagesdir = all_path('JPEGImages/')
vocannotationdir = all_path('Annotations/')


if shutil.os.path.exists(all_path('Annotations')):
    shutil.rmtree(all_path('Annotations'))
if shutil.os.path.exists(all_path('ImageSets')):
    shutil.rmtree(all_path('ImageSets'))
if shutil.os.path.exists(all_path('JPEGImages')):
    shutil.rmtree(all_path('JPEGImages'))

shutil.os.mkdir(all_path('Annotations'))
shutil.os.makedirs(all_path('ImageSets/Main'))
shutil.os.mkdir(all_path('JPEGImages'))

def convertimgset(img_set):
    imgdir = rootdir + "/WIDER_" + img_set + "/images"
    gtfilepath = rootdir + "/wider_face_split/wider_face_" + img_set + "_bbx_gt.txt"
    index=1
    f_set = open(rootdir + "/ImageSets/Main/" + img_set + ".txt", 'w')
    with open(gtfilepath, 'r') as gtfile:
            while (True):  # and len(faces)<10
                filename = gtfile.readline()[:-1]
                if (filename == ""):
                    break
                sys.stdout.write("\r" + str(index) + ":" + filename + "\t\t\t")
                sys.stdout.flush()
                imgpath = imgdir + "/" + filename
                img = cv2.imread(imgpath)
                if not img.data:
                    break

                saveimg = img.copy()
                numbbox = int(gtfile.readline())
                bboxes = []
                for i in range(numbbox):
                    line = gtfile.readline()
                    line = line.split()

                    if (int(line[3]) <= 0 or int(line[2]) <= 0):
                        continue
                    x = int(line[0])
                    y = int(line[1])
                    width = int(line[2])
                    height = int(line[3])
                    bbox = (x, y, width, height)
                    x2 = x + width
                    y2 = y + height

                    if width >= minsize2select and height >= minsize2select and int(line[7])==0 and int(line[4])!=2 and int(line[8])!=2:
                        bboxes.append(bbox)
                    else:
                        saveimg[y:y2, x:x2, :] = (104,117,123)

                # filename = 'r_'+filename.replace("/", "_")
                filename = filename.replace("/", "_")
                if len(bboxes) == 0:
                    print("warrning: no face")
                    continue
                cv2.imwrite(imagesdir + "/" + filename, saveimg)
                # generate filelist
                imgfilepath = filename[:-4]
                f_set.write(imgfilepath + '\n')
                if convert2vocformat:
                    xmlpath = vocannotationdir + "/" + filename
                    xmlpath = xmlpath[:-3] + "xml"
                    doc = Document()
                    annotation = doc.createElement('annotation')
                    doc.appendChild(annotation)
                    folder = doc.createElement('folder')
                    folder_name = doc.createTextNode('widerface')
                    folder.appendChild(folder_name)
                    annotation.appendChild(folder)
                    filenamenode = doc.createElement('filename')
                    filename_name = doc.createTextNode(filename)
                    filenamenode.appendChild(filename_name)
                    annotation.appendChild(filenamenode)
                    source = doc.createElement('source')
                    annotation.appendChild(source)
                    database = doc.createElement('database')
                    database.appendChild(doc.createTextNode('wider face Database'))
                    source.appendChild(database)
                    annotation_s = doc.createElement('annotation')
                    annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
                    source.appendChild(annotation_s)
                    image = doc.createElement('image')
                    image.appendChild(doc.createTextNode('flickr'))
                    source.appendChild(image)
                    flickrid = doc.createElement('flickrid')
                    flickrid.appendChild(doc.createTextNode('-1'))
                    source.appendChild(flickrid)
                    owner = doc.createElement('owner')
                    annotation.appendChild(owner)
                    flickrid_o = doc.createElement('flickrid')
                    flickrid_o.appendChild(doc.createTextNode('yanyu'))
                    owner.appendChild(flickrid_o)
                    name_o = doc.createElement('name')
                    name_o.appendChild(doc.createTextNode('yanyu'))
                    owner.appendChild(name_o)
                    size = doc.createElement('size')
                    annotation.appendChild(size)
                    width = doc.createElement('width')
                    width.appendChild(doc.createTextNode(str(saveimg.shape[0])))
                    height = doc.createElement('height')
                    height.appendChild(doc.createTextNode(str(saveimg.shape[1])))
                    depth = doc.createElement('depth')
                    depth.appendChild(doc.createTextNode(str(saveimg.shape[2])))
                    size.appendChild(width)
                    size.appendChild(height)
                    size.appendChild(depth)
                    segmented = doc.createElement('segmented')
                    segmented.appendChild(doc.createTextNode('0'))
                    annotation.appendChild(segmented)
                    for i in range(len(bboxes)):
                        bbox = bboxes[i]
                        objects = doc.createElement('object')
                        annotation.appendChild(objects)
                        object_name = doc.createElement('name')
                        object_name.appendChild(doc.createTextNode('faceup'))
                        objects.appendChild(object_name)
                        pose = doc.createElement('pose')
                        pose.appendChild(doc.createTextNode('Unspecified'))
                        objects.appendChild(pose)
                        truncated = doc.createElement('truncated')
                        truncated.appendChild(doc.createTextNode('1'))
                        objects.appendChild(truncated)
                        difficult = doc.createElement('difficult')
                        difficult.appendChild(doc.createTextNode('0'))
                        objects.appendChild(difficult)
                        bndbox = doc.createElement('bndbox')
                        objects.appendChild(bndbox)
                        xmin = doc.createElement('xmin')
                        xmin.appendChild(doc.createTextNode(str(bbox[0])))
                        bndbox.appendChild(xmin)
                        ymin = doc.createElement('ymin')
                        ymin.appendChild(doc.createTextNode(str(bbox[1])))
                        bndbox.appendChild(ymin)
                        xmax = doc.createElement('xmax')
                        xmax.appendChild(doc.createTextNode(str(bbox[0] + bbox[2])))
                        bndbox.appendChild(xmax)
                        ymax = doc.createElement('ymax')
                        ymax.appendChild(doc.createTextNode(str(bbox[1] + bbox[3])))
                        bndbox.appendChild(ymax)

                        # #左转90
                        # xmin = doc.createElement('xmin')
                        # xmin.appendChild(doc.createTextNode(str(bbox[1])))
                        # bndbox.appendChild(xmin)
                        # ymin = doc.createElement('ymin')
                        # ymin.appendChild(doc.createTextNode(str(saveimg.shape[1]-bbox[2]-bbox[0])))
                        # bndbox.appendChild(ymin)
                        # xmax = doc.createElement('xmax')
                        # xmax.appendChild(doc.createTextNode(str(bbox[3]+bbox[1])))
                        # bndbox.appendChild(xmax)
                        # ymax = doc.createElement('ymax')
                        # ymax.appendChild(doc.createTextNode(str(saveimg.shape[1] -bbox[0])))
                        # bndbox.appendChild(ymax)

                        # #右转90
                        # xmin = doc.createElement('xmin')
                        # xmin.appendChild(doc.createTextNode(str(saveimg.shape[0]-bbox[1]-bbox[3])))
                        # bndbox.appendChild(xmin)
                        # ymin = doc.createElement('ymin')
                        # ymin.appendChild(doc.createTextNode(str(bbox[0])))
                        # bndbox.appendChild(ymin)
                        # xmax = doc.createElement('xmax')
                        # xmax.appendChild(doc.createTextNode(str(saveimg.shape[0]-bbox[1])))
                        # bndbox.appendChild(xmax)
                        # ymax = doc.createElement('ymax')
                        # ymax.appendChild(doc.createTextNode(str(bbox[0]+bbox[2])))
                        # bndbox.appendChild(ymax)

                        # # 倒置180
                        # xmin = doc.createElement('xmin')
                        # xmin.appendChild(doc.createTextNode(str(saveimg.shape[1] - bbox[0] - bbox[2])))
                        # bndbox.appendChild(xmin)
                        # ymin = doc.createElement('ymin')
                        # ymin.appendChild(doc.createTextNode(str(saveimg.shape[0] - bbox[1] - bbox[3])))
                        # bndbox.appendChild(ymin)
                        # xmax = doc.createElement('xmax')
                        # xmax.appendChild(doc.createTextNode(str(saveimg.shape[1] - bbox[0] )))
                        # bndbox.appendChild(xmax)
                        # ymax = doc.createElement('ymax')
                        # ymax.appendChild(doc.createTextNode(str(saveimg.shape[0] - bbox[1])))
                        # bndbox.appendChild(ymax)

                    f = open(xmlpath, "w")
                    f.write(doc.toprettyxml(indent=''))
                    f.close()
                #     cv2.imshow("img",showimg)
                # cv2.waitKey()
                index = index + 1
    f_set.close()


def rotate_image(path,type):
    for i in os.listdir(path):
        im = Image.open(path+i)
        if type==1:
            im = im.transpose(Image.ROTATE_90)
            im.save(path+i)
        if type==2:
            im = im.transpose(Image.ROTATE_270)
            im.save(path + i)
        if type==3:
            im = im.transpose(Image.ROTATE_180)
            im.save(path + i)



if __name__ == "__main__":
    convertimgset('train')
    # rotate_image(imagesdir,2)
