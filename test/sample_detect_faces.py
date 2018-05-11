import os
import sys
import time

import cv2
from img_utils.files import images_in_dir, filename

from fr.dlibs import detect_faces
from fr.utils import to_rectangle


def main(images_dir, output_dir):
    image_files = images_in_dir(images_dir=images_dir)
    for im_f in image_files:
        im = cv2.imread(im_f)
        start = time.time()
        faces = detect_faces(im)
        end = time.time()
        print('{}: {}'.format(im_f, end - start))

        for face in faces:
            x1, y1, x2, y2 = to_rectangle(face)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(os.path.join(output_dir, filename(im_f)), im)


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 3:
        print('---------------------------------------------------------------------------')
        print('|-- Usage: python test_detect_faces.py ${test_data_dir} ${output_dir}')
        print('---------------------------------------------------------------------------')
        exit(0)
    main(sys.argv[1], sys.argv[2])
