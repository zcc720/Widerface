import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

# os.chdir('E:/code/Widerface/widerface/')
path = 'E:/wider_rotate/up_down_left_right/'

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + 'Annotations/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    xml_path = path
    xml_df = xml_to_csv(xml_path)
    xml_df.to_csv('wider_udlr.csv')
    print('Successfully converted xml to csv.')

main()