# -*- coding: utf-8 -*-
# @Time    : 2019/2/2 13:32
# @Author  : zcc
# @File    : imgaug_BoundingBoxes.py
# @Function:用于数据增强

import xml.etree.ElementTree as ET

import imgaug as ia
from PIL import Image
import numpy as np
import os
from imgaug import augmenters as iaa

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

#读取xml中原bounding boxes坐标
def real_xml(path):
    in_file=open(path)
    tree=ET.parse(in_file)
    root=tree.getroot()
    bndboxlist=[]

    for obj in root.findall('object'):
        bndbox=obj.find('bndbox')
        xmin=int(bndbox.find('xmin').text)
        xmax=int(bndbox.find('xmax').text)
        ymin=int(bndbox.find('ymin').text)
        ymax=int(bndbox.find('ymax').text)

        bndboxlist.append([xmin,ymin,xmax,ymax])

    return bndboxlist

#传入目标变换后的bounding boxe坐标，将原坐标替换成新坐标并生成新的xml文件
def change_xml(dir,image_id,new_target,saveroot,id):
    in_file=open(os.path.join(dir,image_id))
    tree=ET.parse(in_file)
    root=tree.getroot()
    index=0

    for obj in root.findall('object'):
        bndbox=obj.find('bndbox')

        new_xmin=new_target[index][0]
        new_ymin=new_target[index][1]
        new_xmax=new_target[index][2]
        new_ymax=new_target[index][3]

        xmin=bndbox.find('xmin')
        xmin.text=str(new_xmin)
        xmax=bndbox.find('xmax')
        xmax.text=str(new_xmax)
        ymin=bndbox.find('ymin')
        ymin.text=str(new_ymin)
        ymax=bndbox.find('ymax')
        ymax.text=str(new_ymax)

        index=index+1

    tree.write(os.path.join(saveroot,str(image_id[:-4])+'_aug_'+str(id)+'.xml'))


if __name__ == '__main__':
    IMG_DIR = "test_voc/JPEGImages"  # 输入的影像文件夹路径
    XML_DIR = "test_voc/Annotations"  # 输入的XML文件夹路径

    AUG_XML_DIR = "test_voc/AUG_XML"  # 存储增强后的XML文件夹路径
    mkdir(AUG_XML_DIR)

    AUG_IMG_DIR = "test_voc/AUG_IMG"  # 存储增强后的影像文件夹路径
    mkdir(AUG_IMG_DIR)

    AUGLOOP = 5  # 每张影像增强的数量


    new_bndbox=[]
    new_bndbox_list=[]

    #影像增强
    seq=iaa.Sequential([
        # iaa.Flipud(0.5),  #50%概率镜像
        # iaa.Fliplr(0.5),  #50%概率上下翻转
        # iaa.GaussianBlur(sigma=(0,3.0)),
        iaa.Crop(percent=(0, 0.2)),  # random crops
        iaa.Sharpen(alpha=(0, 1.0), ), #
        # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  #浮雕
        iaa.Dropout((0.0, 0.1), per_channel=0.5),
        iaa.Dropout((0.0, 0.1), per_channel=0.5),
        iaa.Add((-10, 10), per_channel=0.5),
        # iaa.Affine(
        #     # translate_px={'x':15,'y':15},
        #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        #     rotate=(-30,30),
        #     shear=(-8, 8)
        # ),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.For the other 50% of all images, we sample the noise per pixel AND channel. This can change the color (not only brightness) of the pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        iaa.ContrastNormalization((0.75, 1.5)), # Strengthen or weaken the contrast in each image.
        # Make some images brighter and some darker.In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
    ])

    for root,sub_folders,files in os.walk(XML_DIR):
        for name in files:
            bndbox=real_xml(os.path.join(XML_DIR,name))

            for epoch in range(AUGLOOP):
                # 保持坐标和图像同步改变，而不是随机
                seq_det=seq.to_deterministic()

                #读取图片
                img=Image.open(os.path.join(IMG_DIR,name.replace('.xml','.jpg')))
                img=np.array(img)

                #bndbox 坐标增强
                for i in range(len(bndbox)):
                    bbs=ia.BoundingBoxesOnImage(
                        [ia.BoundingBox(x1=bndbox[i][0],y1=bndbox[i][1],x2=bndbox[i][2],y2=bndbox[i][3])],
                        shape=img.shape)
                    #变换和记录bounding box
                    bbs_agu=seq_det.augment_bounding_boxes([bbs])[0]


                    # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                    new_bndbox_list.append([int(bbs_agu.bounding_boxes[0].x1),
                                            int(bbs_agu.bounding_boxes[0].y1),
                                            int(bbs_agu.bounding_boxes[0].x2),
                                            int(bbs_agu.bounding_boxes[0].y2)])


                # 变换图像
                image_aug=seq_det.augment_images([img])[0]

                #存储变化后的图片
                path=os.path.join(AUG_IMG_DIR,str(name[:-4])+'_aug_'+str(epoch)+'.jpg')
                Image.fromarray(image_aug).save(path)

                # 存储变化后的XML
                change_xml(XML_DIR, name.replace('.jpg','.xml'), new_bndbox_list,AUG_XML_DIR,epoch)
                print(str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                new_bndbox_list = []







