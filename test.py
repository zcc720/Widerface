# import tensorflow as tf
# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print(sess.run(hello))

# import matplotlib.pyplot as plt
# from PIL import Image
# path='E:\code\Widerface\widerface\JPEGImages\\0--Parade_0_Parade_Parade_0_628.jpg'
# image = Image.open(path)
# plt.imshow(image)
# plt.show()
#
# import os
# path='E:\code\Widerface\widerface\d180\Annotations'
# image_path='E:\code\Widerface\widerface\d180\JPEGImages'
# image_list=[i.split('.')[0] for i in os.listdir(image_path)]
# print(image_list)
# count=0
# for i in os.listdir(path):
#     count+=1
#     if i.split('.')[0]  in image_list:
#         print(i)
# print(count)

import Augmentor
p=Augmentor.Pipeline("./test_image")

# p.rotate(probability=1,max_left_rotation=25,max_right_rotation=25)
p.skew_tilt(probability=1,magnitude=1)
p.sample(30)


import numpy as np
import matplotlib.pyplot as plt
from skimage.io import *
from skimage.transform import *
import scipy.ndimage as ndi


img = imread('test_image/0--Parade_0_Parade_marchingband_1_353.jpg')
print(img.shape)
# 修改图片尺寸（resize）
resized_image = resize(img,(1024,1280))
print(resized_image.shape)

# plt.imshow(resized_image)

#按比例缩放（rescale）
rescaled_img = rescale(img,[0.6,0.5])
print(rescaled_img.shape)
imshow(rescaled_img)
plt.show()

#随机生成5000个椒盐噪声
height,weight,channel = img.shape
img1=img.copy()
for i in range(5000):
    x = np.random.randint(0,height)
    y = np.random.randint(0,weight)
    img1[x ,y ,:] = 255

plt.imshow(img1)
plt.show()

#垂直翻转
vertical_flip = img[::-1,:,:]
imshow(vertical_flip)
plt.show()

#水平翻转
horizontal_flip = img[:,::-1,:]
imshow(horizontal_flip)
plt.show()

#旋转
rotate_img = rotate(img,30)#逆时针旋转30°
imshow(rotate_img)
plt.show()



def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix
def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.
    应用矩阵指定的图像变换。
    # Arguments
    参数
        x: 2D numpy array, single image.
        x: 二维数字阵列，单个图像。
        transform_matrix: Numpy array specifying the geometric transformation.
        transform_matrix: 指定几何变换的Numpy数组。
        channel_axis: Index of axis for channels in the input tensor.
        channel_axis: 输入张量中的通道的轴的索引。
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        fill_mode: 输入的边界之外的点根据给定的模式（{常数”、“最近”、“反射”、“包装”}中的一个填充。
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        cval: 如果mode='constant'，则用于输入边界以外的点的值。
    # Returns
    返回
        The transformed version of the input.
        输入的转换版本。
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=1,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

#原图大小旋转
def rotate(x, theta, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

rotate_limit=(-30, 30)
theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1]) #逆时针旋转角度
#rotate_limit= 30 #自定义旋转角度
#theta = np.pi /180 *rotate_limit #将其转换为PI
img_rot = rotate(img, theta)
plt.imshow(img_rot)
plt.show()

#分别向左和向上偏移原尺寸的0.1倍
def shift(x, wshift, hshift, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    h, w = x.shape[row_axis], x.shape[col_axis] #读取图片的高和宽
    tx = hshift * h #高偏移大小，若不偏移可设为0，若向上偏移设为正数
    ty = wshift * w #宽偏移大小，若不偏移可设为0，若向左偏移设为正数
    translation_matrix = np.array([[1, 0, tx],
                                  [0, 1, ty],
                                  [0, 0, 1]])
    transform_matrix = translation_matrix
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

w_limit=(-0.2, 0.2)
h_limit=(-0.2, 0.2)
wshift = np.random.uniform(w_limit[0], w_limit[1])
hshift = np.random.uniform(h_limit[0], h_limit[1])
#wshift = 0.1 #自定义平移尺寸
#hshift = 0.1 #自定义平移尺寸

img_shift = shift(img, wshift, hshift)
plt.imshow(img_shift)
plt.show()

def zoom(x, zx, zy, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w) #保持中心坐标不改变
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

zoom_range=(0.7, 1)
zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)

#zx = 0.5
#zy = 0.5 #自定义zoom尺寸
img_zoom = zoom(img, zx, zy)
plt.imshow(img_zoom)
plt.show()

#带角度剪切
def shear(x, shear, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                            [0, np.cos(shear), 0],
                            [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

intensity = 0.5
sh = np.random.uniform(-intensity, intensity) #逆时针方向剪切强度为正
img_shear = shear(img, sh)

plt.imshow(img_shear)
plt.show()

#对比度变换，图像的HSV颜色空间，改变H，S和V亮度分量，增加光照变化
from skimage import color

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                            sat_shift_limit=(-255, 255),
                            val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        img = color.rgb2hsv(image)
        h, s ,v = img[:,:,0],img[:,:,1],img[:,:,2]
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])

        h = h + hue_shift

        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = s + sat_shift

        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = v + val_shift

        img[:,:,0],img[:,:,1],img[:,:,2] = h, s ,v

        image = color.hsv2rgb(img)

    return image

contrast_img = randomHueSaturationValue(img)
plt.imshow(contrast_img)
plt.show()

#随机通道偏移（channel shift）
def random_channel_shift(x, intensity, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x
img_chsh = random_channel_shift(img, intensity = 0.05)

plt.imshow(img_chsh)
plt.show()

#PCA
def RGB_PCA(images):
    pixels = images.reshape(-1, images.shape[-1])
    idx = np.random.random_integers(0, pixels.shape[0], 1000000)
    pixels = [pixels[i] for i in idx]
    pixels = np.array(pixels, dtype=np.uint8).T
    m = np.mean(pixels)/256.
    C = np.cov(pixels)/(256.*256.)
    l, v = np.linalg.eig(C)
    return l, v, m


def RGB_variations(image, eig_val, eig_vec):
    a = np.random.randn(3)
    v = np.array([a[0]*eig_val[0], a[1]*eig_val[1], a[2]*eig_val[2]])
    variation = np.dot(eig_vec, v)
    return image + variation

l,v,m = RGB_PCA(img)
img_pca = RGB_variations(img,l,v)
plt.imshow(img_pca)
plt.show()