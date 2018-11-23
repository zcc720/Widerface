import os
import shutil

termi_path='E:/wider_rotate/up_down_left_right/'
def move_file(path):
    for i in os.listdir(path+'Annotations/'):
        shutil.move(path+'Annotations/'+i, termi_path+'Annotations/'+i)
    for i in os.listdir(path + 'JPEGImages/'):
        shutil.move(path+'JPEGImages/'+i, termi_path+'JPEGImages/'+i)

if __name__ == '__main__':
    # move_file('E:/wider_rotate/d180/')
    # move_file('E:/wider_rotate/letf90/')
    # move_file('E:/wider_rotate/right90/')
    move_file('E:/wider_rotate/up/')


