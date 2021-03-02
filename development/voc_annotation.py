import xml.etree.ElementTree as ET
from os import getcwd

# sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets = [('2077', 'trainval'), ('2077', 'test')]

classes = ["warning sign",
           "no reflective cloth",
           "reflective cloth",
           "staircase",
           "insulating tool",
           "tool"]


def convert_annotation(year, image_id, list_file):
    try:
        in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml' % (year, image_id),
                       encoding='utf-8')
    except Exception:
        print('Skip this image')
    else:
        tree = ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            # ################################################
            # pay attention to these codes
            # if you want to train objects that are difficult
            # to be detected, just annotate these codes
            if cls not in classes or int(difficult) == 1:
                continue
            # ################################################
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            print(image_id)
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
                 int(xmlbox.find('ymax').text))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        list_file.write('\n')


wd = getcwd()

for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, image_set),
                     encoding='utf-8').read().strip().split()
    list_file = open('%s_%s.txt' % (year, image_set), 'w')
    for image_id in image_ids:

        image_id_split = image_id.split('/')[-1]

        try:
            in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml' % (year, image_id_split))
        except Exception:
            print('can not find this image')
        else:
            list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (wd, year, image_id))

        convert_annotation(year, image_id_split, list_file)

    list_file.close()
