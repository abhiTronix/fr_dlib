import glob
import os
import sys
import time

import cv2
from img_utils.files import images_in_dir

from fr import detect_faces
from fr.samples_reader import read_from_hierarchy
from fr.utils import to_rectangle

BASE_DIR = os.path.dirname(__file__)


def _subdir(output_dir, label):
    dir_name = os.path.join(output_dir, label)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    return dir_name


def main(samples_dir, output_dir, image_size=160, margin=32):
    hierarchy = read_from_hierarchy(samples_dir)

    for k in hierarchy.keys():
        images_path = hierarchy[k]
        image_files = images_in_dir(images_path)
        subdir = _subdir(output_dir, k)
        start = time.time()
        for im_f in image_files:
            im = cv2.imread(im_f)
            height, width = im.shape[:2]
            faces = detect_faces(im)

            print('{}, {} faces detected'.format(im_f, len(faces)))
            for face in faces:
                images_count = len(glob.glob('{}/*.jpg'.format(subdir)))
                f_name = '{}_{}.jpg'.format(k, '{0:04d}'.format(images_count))
                output_path = os.path.join(subdir, f_name)
                x1, y1, x2, y2 = to_rectangle(face)

                x1 = max(x1 - margin, 0)
                y1 = max(y1 - margin, 0)
                x2 = min(x2 + margin, width)
                y2 = min(y2 + margin, height)

                roi = im[y1:y2, x1:x2, :]

                roi = cv2.resize(roi, (image_size, image_size))
                cv2.imwrite(output_path, roi)

        print('{} done, time spent: {}'.format(k, time.time() - start))


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 3:
        print('---------------------------------------------------------------------------')
        print('|-- Usage: python sample_alignment.py ${samples_dir} ${output_dir}')
        print('---------------------------------------------------------------------------')
        exit(0)
    main(sys.argv[1], sys.argv[2])
