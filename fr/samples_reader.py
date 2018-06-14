# -*- coding: utf-8 -*-
import os
import pickle

import cv2
from img_utils.files import images_in_dir

from . import detect_faces, compute_face_descriptor


def read_from_hierarchy(images_dir):
    """

    :param images_dir: images dir which is organized as below
                     ├── Ariel_Sharon
                     │   ├── Ariel_Sharon_0006.png
                     │   ├── Ariel_Sharon_0007.png
                     │   ├── Ariel_Sharon_0008.png
                     │   ├── Ariel_Sharon_0009.png
                     │   └── Ariel_Sharon_0010.png
                     |
                     ├── Arnold_Schwarzenegger
                     │   ├── Arnold_Schwarzenegger_0006.png
                     │   ├── Arnold_Schwarzenegger_0007.png
                     │   ├── Arnold_Schwarzenegger_0008.png
                     │   ├── Arnold_Schwarzenegger_0009.png
                     │   └── Arnold_Schwarzenegger_0010.png
                     |
                     ├── Colin_Powell
                     │   ├── Colin_Powell_0006.png
                     │   ├── Colin_Powell_0007.png
    :return:    dict that key is name of each image, value is image files list such as
                {'Ariel_Sharon': '/data/Ariel_Sharon/' }
    """
    sub_dirs = [x for x in os.walk(images_dir)]
    hierarchy = {}
    for d in sub_dirs[0][1]:
        hierarchy[d] = os.path.join(images_dir, d)
    return hierarchy


def load_samples_descriptors(samples_dir):
    """
    :param samples_dir: images dir which is organized as below

                     samples_dir/
                     ├── Ariel_Sharon
                     │   ├── Ariel_Sharon_0006.png
                     │   ├── Ariel_Sharon_0007.png
                     │   ├── Ariel_Sharon_0008.png
                     │   ├── Ariel_Sharon_0009.png
                     │   └── Ariel_Sharon_0010.png
                     |
                     ├── Arnold_Schwarzenegger
                     │   ├── Arnold_Schwarzenegger_0006.png
                     │   ├── Arnold_Schwarzenegger_0007.png
                     │   ├── Arnold_Schwarzenegger_0008.png
                     │   ├── Arnold_Schwarzenegger_0009.png
                     │   └── Arnold_Schwarzenegger_0010.png
                     |
                     ├── Colin_Powell
                     │   ├── Colin_Powell_0006.png
                     │   ├── Colin_Powell_0007.png

    :return: two lists, one for face encodings, one for corresponding class names
    """
    images_dict = read_from_hierarchy(images_dir=samples_dir)
    print(len(images_dict), images_dict)
    class_names = []
    face_descriptors = []
    for k in images_dict.keys():
        images_dir = images_dict[k]
        image_files = images_in_dir(images_dir)
        print('{}: {}'.format(k, image_files))
        for im_f in image_files:
            im = cv2.imread(im_f)
            faces = detect_faces(im)
            for face in faces:
                descriptor = compute_face_descriptor(im, face)
                face_descriptors.append(descriptor)
                class_names.append(k)

    return face_descriptors, class_names


def load_from_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data['face_descriptors'], data['class_names']


def save2pickle(face_descriptors, class_names, pickle_path):
    data = {
        'face_descriptors': face_descriptors,
        'class_names': class_names
    }
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
